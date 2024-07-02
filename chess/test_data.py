import os
os.environ["ROCM_PATH"] = "/opt/rocm"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import numpy as np
import matplotlib.pyplot as plt
import chess
import torch
import math
from utils import *
from chess_net import ChessModel
from supervised_data import position_reader, game_reader, generate_positions, get_positions
from torch.utils.data import DataLoader
import time
from timeit import default_timer as timer
from torch.utils.data.datapipes.iter import IterableWrapper, Shuffler  # type: ignore
import train

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
batch_size = 4096

def benchmark_data(loader : DataLoader):
    start = timer()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
    train.train_loop(loader,model,policy_loss_fn,value_loss_fn,optimizer,0, save=False, print_counter=1)
    print(f"Elapsed time: {timer() - start}")
    
def benchmark_iterator(itr):
    start = timer()
    for x in itr:
        pass
    print(f"Elapsed time: {timer() - start}")
    
def loader1():
    data = position_reader("/home/gerard/Documents/Personal/Programming/rl/chess/data/lichess_2013/shortened_data.pgn")
    data = IterableWrapper(data,deepcopy=False)
    data = data.shuffle(buffer_size=batch_size*4)
    data = data.sharding_filter()
    mapped = data.map(get_position_rep)
    loader = DataLoader(dataset=mapped, batch_size=batch_size, num_workers=8,pin_memory=True)   
    return loader 

def loader2():
    data = game_reader("/home/gerard/Documents/Personal/Programming/rl/chess/data/lichess_2013/shortened_data.pgn")
    data = IterableWrapper(data, deepcopy=False)
    data = data.sharding_filter()
    data = data.map(get_positions)
    data = data.unbatch()
    data = data.map(get_position_rep)
    data = data.shuffle(buffer_size=batch_size*4)
    loader = DataLoader(dataset=data, batch_size=batch_size, num_workers=8,pin_memory=True)   
    return loader

    
    
if __name__ == "__main__":
    model = ChessModel().to(device)
    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()
    # benchmark_iterator(position_reader("/home/gerard/Documents/Personal/Programming/rl/chess/data/lichess_2013/shortened_data.pgn"))
    print("Benchmarking dataloader 2")
    # benchmark_iterator(game_reader("/home/gerard/Documents/Personal/Programming/rl/chess/data/lichess_2013/shortened_data.pgn"))
    benchmark_data(loader2())
    print("Benchmarking dataloader 1")
    benchmark_data(loader1())

    