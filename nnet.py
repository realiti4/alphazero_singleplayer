import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import (argmax,check_space,is_atari_game,copy_atari_state,store_safely,
restore_atari_state,stable_normalizer,smooth,symmetric_remove,Database)


class NNet(nn.Module):
    def __init__(self, Env, n_hidden_layers, n_hidden_units):
        super(NNet, self).__init__()

        self.action_dim, self.action_discrete  = check_space(Env.action_space)
        self.state_dim, self.state_discrete  = check_space(Env.observation_space)
        if not self.action_discrete: 
            raise ValueError('Continuous action space not implemented')

        self._hidden_layers = n_hidden_layers
        self._hidden_units = n_hidden_units

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)

        self.fc3 = nn.Linear(128, 2)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.softmax(pi, dim=1), v

        return F.log_softmax(pi, dim=1), torch.tanh(v)