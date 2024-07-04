import os
os.environ["ROCM_PATH"] = "/opt/rocm"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import numpy as np

import matplotlib.pyplot as plt
import chess
import math
import torch
import utils
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import random


class BasicBlock(nn.Module):
    def __init__(self,  channels = 256):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels,kernel_size=3,padding='same'),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels,kernel_size=3,padding='same'),
            nn.BatchNorm2d(channels))
        self.relu = nn.ReLU()
    def forward(self,x):
        identity = x
        out = self.block(x)
        out += identity
        out = self.relu(out)
        return out

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
            self.build_conv_layer(20,256),
            self.build_residual_block(256),
            self.build_residual_block(256),
            self.build_residual_block(256),
            self.build_residual_block(256),
            self.build_residual_block(256),
            self.build_residual_block(256),
            self.build_residual_block(256),
            self.build_residual_block(256),
            self.build_residual_block(256),
            self.build_residual_block(256),
            self.build_residual_block(256)
        )
    def build_residual_block(self,channels):
        return BasicBlock(channels)
    def build_conv_layer(self, in_channels, out_channels) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=3,padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def build_policy_head(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16,kernel_size=1,padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*8*16,np.prod([*self.policy_shape]))
        )
    def build_value_head(self):
        return nn.Sequential(
            nn.Conv2d(256, 4,kernel_size=1,padding='same'),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8*8*4,1),
            nn.Tanh()
        )
    def forward(self, x):
        rep = self.representation(x)
        return self.policy_head(rep), self.value_head(rep)
    
if __name__ == "__main__":
    model = ChessModel().to('cuda')
    state_dict = torch.load("/home/gerard/Documents/Personal/Programming/rl/chess/checkpoints/1/checkpoint_1_1600.pt")['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    print(model)
    board = chess.Board()
    # for j in range(21):
    #     moves = list(board.generate_legal_moves())
    #     move = np.random.choice(a=moves)
    #     board.push(move)
    chess.Board.set_fen(board, "r2qk2r/pbp1bppp/1pnp1n2/1B2p1B1/3P4/2N1PN2/PPP2PPP/R2QK2R w KQkq - 2 8")
    print(board)
    print(board.fen())
    board_rep = utils.get_board_rep(board)
    board_rep_tensor = torch.tensor(board_rep).to('cuda', torch.float32)
    p, v = model(board_rep_tensor.unsqueeze(0))
    print(p,v,p.shape,v.shape)
    move = utils.sample_move(p)
    utils.show_move_rep(torch.tensor(move))
    utils.show_move_rep(p)
    # for i in range(100):
    #     board = chess.Board()
    #     for j in range(20):
    #         moves = list(board.generate_legal_moves())
    #         move = np.random.choice(a=moves)
    #         board.push(move)
    #     print(board)
    #     board_rep = utils.get_board_rep(board)
    #     board_rep_tensor = torch.tensor(board_rep).to('cuda', torch.float32)
    #     p, v = model(board_rep_tensor.unsqueeze(0))
    #     print(v)