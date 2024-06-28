import os
os.environ["ROCM_PATH"] = "/opt/rocm"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import numpy as np

import matplotlib.pyplot as plt
import chess
import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


class ChessModel(nn.Module):
    def __init__(self,  hidden_layers=1, filters=1, input_shape = (20,8,8), policy_shape =(76,8,8)):
        super().__init__()
        self.input_shape = input_shape
        self.policy_shape = policy_shape
        self.hidden_layers = hidden_layers
        self.filters = filters
        self.representation = self.build_representation_model()
        self.policy_head = self.build_policy_head()
        self.value_head = self.build_value_head()
    def build_representation_model(self):
        return nn.Sequential(
            self.build_conv_layer(20,30),
            self.build_conv_layer(30,50),
            self.build_conv_layer(50,70),
            self.build_conv_layer(70,100)
        )
    def build_conv_layer(self, in_channels, out_channels) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=3,padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def build_policy_head(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=2,kernel_size=1,padding='same'),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128,np.prod([*self.policy_shape]))
        )
    def build_value_head(self):
        return nn.Sequential(
            nn.Conv2d(100, 1,kernel_size=1,padding='same'),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64,1),
            nn.Tanh()
        )
    def forward(self, x):
        rep = self.representation(x)
        return self.policy_head(rep), self.value_head(rep)
    
if __name__ == "__main__":
    model = ChessModel().to('cuda')
    print(model)
    board_rep = np.random.rand(20,8,8)
    board_rep_tensor = torch.tensor(board_rep).to('cuda', torch.float32)
    print(board_rep_tensor.shape)
    p, v = model(board_rep_tensor.unsqueeze(0))
    print(p,v)
    print(p.shape, v.shape)