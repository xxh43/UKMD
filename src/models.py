
import torch.nn as nn
import torch.nn.functional as F
import torch

import argparse
import os
import numpy as np
import math
from config import * 


class JointEncoder(nn.Module):
    def __init__(self):
        super(JointEncoder, self).__init__()

        self.bn_conv1 = nn.Sequential(nn.Conv1d(4, 64, 1),
                            nn.BatchNorm1d(64),
                            nn.LeakyReLU(negative_slope=0.2))

        self.bn_conv2 = nn.Sequential(nn.Conv1d(64, 128, 1),
                            nn.BatchNorm1d(128),
                            nn.LeakyReLU(negative_slope=0.2))
        
        self.bn_conv3 = nn.Sequential(nn.Conv1d(128, 512, 1),
                            nn.BatchNorm1d(512),
                            nn.LeakyReLU(negative_slope=0.2)
                            )
        
        self.fc1 = nn.Sequential(nn.Linear(512, 256),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.fc_enc = nn.Sequential(nn.Linear(256, enc_dim)
                                   )

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.bn_conv1(x)
        x = self.bn_conv2(x)
        x = self.bn_conv3(x)

        x = torch.max(x, 2, keepdim=True)[0]
        x = torch.flatten(x, 1, -1)

        x = self.fc1(x)
        x = self.fc_enc(x)

        return x