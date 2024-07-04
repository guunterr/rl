import time
import lichess.api
import matplotlib.pyplot as plt
import chess
import chess.pgn
import numpy as np
import itertools
import utils
import torch
import torch.utils.data as data
from torch.utils.data.datapipes.iter import IterableWrapper
    
def position_reader(filepath):
    pgn = open(filepath)
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            return
        result = game.headers["Result"]
        while game is not None and game.next() is not None:
            board = game.board()
            move = game.next().move
            player=board.turn
            if result[1] == "/":
                winning = 0
            elif result[0] == "1":
                if player == chess.WHITE:
                    winning = 1
                else:
                    winning = -1
            elif result[0] == "0":
                if player == chess.WHITE:
                    winning = -1
                else:
                    winning = 1
            else:
                raise RuntimeError("Don't know if this game is a win or a draw")
            yield board, move, winning
            game = game.next()
            
def game_reader(filepaths):
    for filepath in filepaths:
        pgn = open(filepath)
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                return
            yield game
        
def get_positions(game: chess.pgn.Game):
    return list(generate_positions(game))

def generate_positions(game: chess.pgn.Game):
    result = game.headers["Result"]
    while game is not None and game.next() is not None:
        board = game.board()
        move = game.next().move
        player=board.turn
        if result[1] == "/":
            winning = 0
        elif result[0] == "1":
            if player == chess.WHITE:
                winning = 1
            else:
                winning = -1
        elif result[0] == "0":
            if player == chess.WHITE:
                winning = -1
            else:
                winning = 1
        else:
            raise RuntimeError("Don't know if this game is a win or a draw")
        yield board, move, winning
        game = game.next()

def dataloader_from_filepaths(filepaths, batch_size, num_workers):
    data = game_reader(filepaths)
    data = IterableWrapper(data, deepcopy=False)
    data = data.sharding_filter()
    data = data.map(get_positions)
    data = data.unbatch()
    data = data.map(utils.get_position_rep)
    data = data.shuffle(buffer_size=batch_size*4)
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, num_workers=num_workers,pin_memory=True)   
    return loader


if __name__ == "__main__":
    pass