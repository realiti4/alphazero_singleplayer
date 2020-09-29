import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nnet import NNet

from helpers import (argmax,check_space,is_atari_game,copy_atari_state,store_safely,
restore_atari_state,stable_normalizer,smooth,symmetric_remove,Database)


class model_dev:
    def __init__(self, Env, n_hidden_layers, n_hidden_units):
        # super(model_dev, self).__init__()

        self.action_dim, self.action_discrete  = check_space(Env.action_space)
        self.state_dim, self.state_discrete  = check_space(Env.observation_space)
        if not self.action_discrete: 
            raise ValueError('Continuous action space not implemented')

        # self._hidden_layers = n_hidden_layers
        # self._hidden_units = n_hidden_units

        self.nnet = NNet(Env, n_hidden_layers, n_hidden_units)

    def train(self, sb, vb, pib):
        # optimizer = optim.RMSprop(self.nnet.parameters(), lr=0.001)
        optimizer = optim.Adam(self.nnet.parameters(), lr=0.001)

        mse = nn.MSELoss()
        cross = nn.CrossEntropyLoss()

        for i in range(10):
            self.nnet.train()

            sb = torch.FloatTensor(sb)

            out_pi, out_v = self.nnet(sb)


            # Loss
            vb = torch.FloatTensor(vb)
            pib = torch.FloatTensor(pib)

            v_loss = mse(out_v, vb)
            # pi_loss = mse(out_pi, pib)
            pi_loss = cross(out_pi, pib.max(1)[0].long())    # TODO fix here

            loss = v_loss + pi_loss     # TODO tf.reduce_mean(self.pi_loss)


            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # print('debug')




    def predict(self, s):
        s = torch.FloatTensor(s)

        self.nnet.eval()

        with torch.no_grad():

            pi, v = self.nnet(s)

        return pi.cpu().numpy()[0], v.cpu().numpy()[0]

        print('debug')

    def predict_V(self, s):
        pass

    def predict_pi(self, s):
        pass