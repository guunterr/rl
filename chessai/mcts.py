import os
os.environ["ROCM_PATH"] = "/opt/rocm"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import numpy as np
import matplotlib.pyplot as plt
import chess
import torch
import math
import warnings
import train
import utils
import random
import chess_net
import parse_data
import time

from torch.utils.data.datapipes.iter import IterableWrapper, Shuffler # type: ignore

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def ucb_score(parent, child):
    prior_score = child.p * math.sqrt(parent.visits) / (child.visits + 1)
    if child.visits > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.w()
    else:
        value_score = 0

    return value_score + prior_score

class MCTSNode():
    def __repr__(self):
        return f"p: {self.p:.4f}, children={len(self.children)}\n"
    def __init__(self, p, board : chess.Board, visited=False):
        self.board = board
        self.visits = 0
        self.children = {}
        self.n = self.w = self.q = 0
        self.p = p
    def visited(self):
        return len(self.children) > 0
    def select_child(self):
        best_score = -np.inf
        best_move = None
        best_child = None
        for move,child in self.children.items():
            score = ucb_score(self,child)
            if score > best_score:
                best_score, best_move, best_child = score, move, child
        return best_move, best_child
    def expand(self, model):
        board_rep = torch.tensor(utils.get_board_rep(self.board)).to(device)
        move_rep, value_rep = model(board_rep.unsqueeze(0))
        move_rep = utils.move_rep_to_numpy(torch.nn.functional.softmax(move_rep, dim=1))
        legal_moves = list(self.board.generate_legal_moves())
        move_probabilities = {}
        for move in legal_moves:
            move_filter = utils.get_move_rep(move)
            move_probabilities[move] = np.sum(np.multiply(move_filter, move_rep.reshape(4864)))
        # utils.show_move_rep(torch.tensor(move_rep), normalize_probabilities=False)
        for move,p in move_probabilities.items():
            new_board = board.copy(stack=False)
            new_board.push(move)
            self.children[move] = MCTSNode(p, new_board)
class MCTSSearch():
    def __init__(self, model, board : chess.Board):
        self.model = model
        self.board = board

    def run(self, num_simulations=20):
        root = MCTSNode(0, self.board.copy(), visited=True)
        root.expand(self.model)
        for i in range(num_simulations):
            node = root
            search_path = [node]
            while node.visited():
                move, node = node.select_child()
                search_path.append(node)
            
            parent = search_path[-2]
            board = parent.board
            next_board = board.copy()
            next_board.push(move)
            outcome = next_board.outcome()
            if outcome is None:
                search_path[-1].expand(self.model)
                value = 0
            else: 
                value = 0 if outcome.winner == None else int(outcome.winner)*2 - 1
            self.backpropagate(search_path, value if value is not None else 0)
        return
    def backpropagate(self, search_path : list[MCTSNode], value):
        for node in reversed(search_path):
            if node.board.turn == self.board.turn:
                node.w += value
            else:
                node.w -= value
            node.visits+=1
        
            
if __name__ == "__main__":   
    model = chess_net.ChessModel()
    model.load_state_dict(torch.load("/home/gerard/Documents/Personal/Programming/rl/chessai/model.pt")['model_state_dict'])
    model.eval()
    model.to(device)
    board = chess.Board()
    for _ in range(15):
        legals = list(board.generate_legal_moves())
        board.push(legals[np.random.randint(0,len(legals))])
    print(board)
    search = MCTSSearch(model, board)
    search.run(num_simulations=5)
    print("DONE!")