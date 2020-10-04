import copy

import numpy as np

from utils.helpers import (argmax, check_space, is_atari_game, copy_atari_state, store_safely,
                restore_atari_state, stable_normalizer, smooth, symmetric_remove, Database)


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
        # pi, v = self.model.predict(self.index[None, ])      # (1, 4)
        pi, v = self.model.predict(self.index.reshape(1, 49))   # Dev
        v = v[0]    # Int
        self.V = v if not self.terminal else np.array(0.0)
        return pi, v    # (2,) and int
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
                s1, r, t, _ = mcts_env.step(action.index)   # (4,), int, False, _
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