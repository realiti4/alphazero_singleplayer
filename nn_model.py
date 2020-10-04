import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nnet import NNet

from utils.helpers import check_space

use_cuda = torch.cuda.is_available()
use_cuda = False

class Model:
    def __init__(self, Env, n_hidden_layers, n_hidden_units):
        # super(model_dev, self).__init__()

        self.action_dim, self.action_discrete  = check_space(Env.action_space)
        self.state_dim, self.state_discrete  = check_space(Env.observation_space)
        if not self.action_discrete: 
            raise ValueError('Continuous action space not implemented')

        # self._hidden_layers = n_hidden_layers
        # self._hidden_units = n_hidden_units

        self.nnet = NNet(Env, n_hidden_layers, n_hidden_units)

        if use_cuda:
            self.nnet.cuda()

    def train(self, sb, vb, pib):
        # optimizer = optim.RMSprop(self.nnet.parameters(), lr=0.001)
        optimizer = optim.Adam(self.nnet.parameters(), lr=0.001)

        mse = nn.MSELoss()
        cross = nn.CrossEntropyLoss()

        vb = torch.FloatTensor(vb)
        pib = torch.FloatTensor(pib)
        sb = torch.FloatTensor(sb)

        if use_cuda:
            sb, vb, pib = sb.cuda(), vb.cuda(), pib.cuda()

        for i in range(10):
            self.nnet.train()   

            out_pi, out_v = self.nnet(sb)

            # Loss          
            v_loss = mse(out_v, vb)
            pi_loss = cross(out_pi, pib.max(1)[0].long())    # TODO fix here

            loss = v_loss + pi_loss     # TODO tf.reduce_mean(self.pi_loss)


            optimizer.zero_grad()

            loss.backward()
            optimizer.step()


    def predict(self, s):
        s = torch.FloatTensor(s)
        if use_cuda:
            s = s.cuda()

        self.nnet.eval()

        with torch.no_grad():
            pi, v = self.nnet(s)

        return pi.cpu().numpy()[0], v.cpu().numpy()[0]

        print('debug')
