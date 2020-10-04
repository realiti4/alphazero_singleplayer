import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class TradingNNet(nn.Module):
    def __init__(self, layers, num_channels):
        super(TradingNNet, self).__init__()
        # self.args = args

        self.conv1 = nn.Conv1d(48, num_channels, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(num_channels)

        self.fc1 = nn.Linear(num_channels, num_channels)     # (512, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc_balance = nn.Linear(1, num_channels)
        self.fc_connect = nn.Linear(2*num_channels, num_channels)

        self.fc3 = nn.Linear(num_channels, 2)
        self.fc4 = nn.Linear(num_channels, 1)

    def forward(self, s):
        if len(s.shape) != 3:
            s = s.unsqueeze(2)
        s, balance = s[:, :-1, :], s[:, -1, :]
        # s, balance = s[:, :-1], s[:, -1]

        # s = F.relu(self.bn1(self.conv1(s)))     # TODO fix bn, it is not accepting 1 feature
        s = F.relu(self.conv1(s))
        s = s.squeeze(-1)

        s = torch.tanh(s)   # tanh might be helping a lot after a conv TODO find out why

        s = F.dropout(self.fc1(s), p=0.4)       # There was a training= variable

        balance = torch.tanh(self.fc_balance(balance))
        s = self.fc_connect(torch.cat((s, balance), dim=1))

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.softmax(pi, dim=1), v

        return F.log_softmax(pi, dim=1), torch.tanh(v)
