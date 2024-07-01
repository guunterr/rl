import time
import lichess.api
import matplotlib.pyplot as plt
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


        
# class PositionDataset(data.IterableDataset):
#     def __init__(self, filename, buffer_size):
#         super().__init__()
#         self.buffer_size = buffer_size
#         self.generator = position_generator(filename)
#     def __iter__(self):
#         return self.generator
        
if __name__ == "__main__":
    path_to_games = "chess/data/lichess_2013/lichess_db_standard_rated_2013-06.pgn"
    gen = position_reader(path_to_games)
    pgn = open(path_to_games)
    count = 0
    start = time.time()
    for pos in gen:
        # rep = utils.get_position_rep(pos)
        if count > 100000:
            print(time.time() - start)
            break
        count += 1
        # print(move_rep(pos[1])[:,:,6])
        # print(pos[1])
        # rep = utils.get_move_rep(pos[1])
        # utils.show_move_rep(rep)
        # break
        # if pos[1].promotion is not None and pos[1].promotion != 5:
        #     print(count, pos[1], chess.piece_name(pos[1].promotion), chess.piece_symbol(pos[1].promotion), pos[1].promotion)
        #     rep = utils.get_move_rep(pos[1])
        #     utils.show_move_rep(rep)
        #     break
        # count+=1
        # if count > 1000000:
        #     break