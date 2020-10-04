"""
One-player Alpha Zero
@author: Thomas Moerland, Delft University of Technology
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import time
import copy
from gym import wrappers
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from helpers import (argmax, check_space, is_atari_game, copy_atari_state, store_safely,
                restore_atari_state, stable_normalizer, smooth, symmetric_remove, Database)
from rl.make_game import make_game

from nn_model import Model


##### MCTS functions #####
      
class Action():
    ''' Action object '''
    def __init__(self, index, parent_state, Q_init=0.0):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0
        self.n = 0
        self.Q = Q_init
                
    def add_child_state(self, s1, r, terminal, model):
        self.child_state = State(s1, r, terminal, self, self.parent_state.na, model)
        return self.child_state
        
    def update(self, R):
        self.n += 1
        self.W += R
        self.Q = self.W / self.n

class State():
    ''' State object '''

    def __init__(self, index, r, terminal, parent_action, na, model):
        ''' Initialize a new state '''
        self.index = index # state
        self.r = r # reward upon arriving in this state
        self.terminal = terminal # whether the domain terminated in this state
        self.parent_action = parent_action
        self.n = 0
        self.model = model
        
        pi, v = self.evaluate()
        # Child actions
        self.na = na
        self.child_actions = [Action(a, parent_state=self, Q_init=self.V) for a in range(na)]
        self.priors = pi
        # self.priors = model.predict_pi(index[None,]).flatten()
    
    def select(self, c=1.5):
        ''' Select one of the child actions based on UCT rule '''
        UCT = np.array([child_action.Q + prior * c * (np.sqrt(self.n + 1) / (child_action.n + 1)) for child_action,prior in zip(self.child_actions,self.priors)]) 
        winner = argmax(UCT)
        return self.child_actions[winner]

    def evaluate(self):
        ''' Bootstrap the state value '''
        pi, v = self.model.predict(self.index[None, ])
        v = v[0]
        self.V = v if not self.terminal else np.array(0.0)
        return pi, v
        # self.V = np.squeeze(self.model.predict_V(self.index[None,])) if not self.terminal else np.array(0.0)          

    def update(self):
        ''' update count on backward pass '''
        self.n += 1
        
class MCTS():
    ''' MCTS object '''

    def __init__(self, root, root_index, model, na, gamma):
        self.root = None
        self.root_index = root_index
        self.model = model
        self.na = na
        self.gamma = gamma
    
    def search(self, n_mcts, c, Env, mcts_env):
        ''' Perform the MCTS search from the root '''
        if self.root is None:
            self.root = State(self.root_index, r=0.0, terminal=False, parent_action=None, na=self.na, model=self.model) # initialize new root
        else:
            self.root.parent_action = None # continue from current root
        if self.root.terminal:
            raise(ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(Env)
        if is_atari:
            snapshot = copy_atari_state(Env) # for Atari: snapshot the root at the beginning     
        
        for i in range(n_mcts):     
            state = self.root # reset to root for new trace
            if not is_atari:
                mcts_env = copy.deepcopy(Env) # copy original Env to rollout from
            else:
                restore_atari_state(mcts_env,snapshot)            
            
            while not state.terminal: 
                action = state.select(c=c)
                s1, r, t, _ = mcts_env.step(action.index)
                if hasattr(action, 'child_state'):
                    state = action.child_state # select
                    continue
                else:
                    state = action.add_child_state(s1, r, t, self.model) # expand
                    break

            # Back-up 
            R = state.V         
            while state.parent_action is not None: # loop back-up until root is reached
                R = state.r + self.gamma * R 
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()                
    
    def return_results(self, temp):
        ''' Process the output at the root node '''
        counts = np.array([child_action.n for child_action in self.root.child_actions])
        Q = np.array([child_action.Q for child_action in self.root.child_actions])
        pi_target = stable_normalizer(counts,temp)
        V_target = np.sum((counts/np.sum(counts))*Q)[None]
        return self.root.index,pi_target,V_target
    
    def forward(self, a, s1):
        ''' Move the root forward '''
        if not hasattr(self.root.child_actions[a],'child_state'):
            self.root = None
            self.root_index = s1
        elif np.linalg.norm(self.root.child_actions[a].child_state.index - s1) > 0.01:
            print('Warning: this domain seems stochastic. Not re-using the subtree for next search. '+
                  'To deal with stochastic environments, implement progressive widening.')
            self.root = None
            self.root_index = s1            
        else:
            self.root = self.root.child_actions[a].child_state



#### Agent ##
def agent(game, n_ep, n_mcts, max_ep_len, lr, c, gamma, data_size, batch_size, temp, n_hidden_layers, n_hidden_units):
    ''' Outer training loop '''
    #tf.reset_default_graph()
    episode_returns = [] # storage
    timepoints = []
    # Environments
    Env = make_game(game)
    is_atari = is_atari_game(Env)
    mcts_env = make_game(game) if is_atari else None

    D = Database(max_size=data_size,batch_size=batch_size)        
    model = Model(Env=Env, n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units)
    t_total = 0 # total steps   
    R_best = -np.Inf
 
    for ep in range(n_ep):    
        start = time.time()
        s = Env.reset() 
        R = 0.0 # Total return counter
        a_store = []
        seed = np.random.randint(1e7) # draw some Env seed
        Env.seed(seed)      
        if is_atari: 
            mcts_env.reset()
            mcts_env.seed(seed)                                

        mcts = MCTS(root_index=s, root=None, model=model, na=model.action_dim, gamma=gamma) # the object responsible for MCTS searches                             
        for t in range(max_ep_len):
            # MCTS step
            mcts.search(n_mcts=n_mcts, c=c, Env=Env, mcts_env=mcts_env) # perform a forward search
            state, pi, V = mcts.return_results(temp) # extract the root output
            D.store((state, V, pi))

            # Make the true step
            a = np.random.choice(len(pi), p=pi)
            a_store.append(a)
            s1, r, terminal, _ = Env.step(a)
            R += r
            t_total += n_mcts # total number of environment steps (counts the mcts steps)                

            if terminal:
                break
            else:
                mcts.forward(a,s1)
        
        # Finished episode
        episode_returns.append(R) # store the total episode return
        timepoints.append(t_total) # store the timestep count of the episode return
        # store_safely(os.getcwd(), 'result', {'R':episode_returns, 't':timepoints})

        if R > R_best:
            a_best = a_store
            seed_best = seed
            R_best = R
        print('Finished episode {}, total return: {}, total time: {} sec'.format(ep, np.round(R, 2), np.round((time.time()-start), 1)))
        # Train
        D.reshuffle()
        for epoch in range(1):
            for sb, Vb, pib in D:
                model.train(sb, Vb, pib)

    # Return results
    return episode_returns, timepoints, a_best, seed_best, R_best

#### Command line call, parsing and plotting ##
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='CartPole-v0', help='Training environment')
    parser.add_argument('--n_ep', type=int, default=500, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=25, help='Number of MCTS traces per step')
    parser.add_argument('--max_ep_len', type=int, default=300, help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--c', type=float, default=1.5, help='UCT constant')
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature in normalization of counts to policy target')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount parameter')
    parser.add_argument('--data_size', type=int, default=1000, help='Dataset size (FIFO)')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--window', type=int, default=25, help='Smoothing window for visualization')

    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers in NN')
    parser.add_argument('--n_hidden_units', type=int, default=128, help='Number of units per hidden layers in NN')

    
    args = parser.parse_args()
    episode_returns, timepoints, a_best, seed_best, R_best = agent(game=args.game, n_ep=args.n_ep, n_mcts=args.n_mcts, 
                                        max_ep_len=args.max_ep_len, lr=args.lr, c=args.c, gamma=args.gamma, 
                                        data_size=args.data_size, batch_size=args.batch_size, temp=args.temp, 
                                        n_hidden_layers=args.n_hidden_layers, n_hidden_units=args.n_hidden_units)

    # Finished training: Visualize
    fig, ax = plt.subplots(1, figsize=[7, 5])
    total_eps = len(episode_returns)
    episode_returns = smooth(episode_returns, args.window, mode='valid') 
    ax.plot(symmetric_remove(np.arange(total_eps), args.window-1), episode_returns, linewidth=4, color='darkred')
    ax.set_ylabel('Return')
    ax.set_xlabel('Episode', color='darkred')
    plt.savefig(os.getcwd()+'/learning_curve.png', bbox_inches="tight", dpi=300)
    
#    print('Showing best episode with return {}'.format(R_best))
#    Env = make_game(args.game)
#    Env = wrappers.Monitor(Env, os.getcwd() + '/best_episode', force=True)
#    Env.reset()
#    Env.seed(seed_best)
#    for a in a_best:
#        Env.step(a)
#        Env.render()
