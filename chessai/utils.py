import chess
import numpy as np
import matplotlib.pyplot as plt
import torch

piece_layer = {"P":0,"R":1,"N":2,"B":3,"Q":4, "K":5,"p":6,"r":7,"n":8,"b":9,"q":10, "k":11}
direction = {0:"left", 1:"fwd", 2:"right"}
move_layer_names = {layer : f"{chess.piece_name(2 + layer // 3)[0]} {direction[layer% 3]}" for layer in range(0,12)}
def get_board_rep(board: chess.Board):
    board_rep = np.zeros((20,8,8),dtype=np.float32)
    pieces = board.board_fen().split("/")

    for (i,rank) in enumerate(pieces):
        j = 0
        while j < 8:
            piece = rank[0]
            rank = rank[1:]
            if piece.isnumeric():
                j += int(piece)
            else:
                board_rep[piece_layer[piece],i,j] = 1
                j+=1
                
    fen = board.fen().split(" ")[1:]
    if fen[0] == "w":
        board_rep[12,:,:] = 1
    if fen[2] != "-":
        i,j = chess.FILE_NAMES.index(fen[2][0]), chess.RANK_NAMES.index(fen[2][1])
        board_rep[13,i,8-j] = 1
    board_rep[14,:,:] = int(fen[3])
    board_rep[15,:,:] = int(fen[4])
    for (n, castle) in enumerate(["K","Q","k","q"]):
        if castle in fen[1]:
            board_rep[16+n,:,:] = 1
    return board_rep

def to_coords(square: chess.Square):
    return 7-chess.square_rank(square),chess.square_file(square)

def from_coords(x, y):
    return chess.square_name(chess.square(x,7-y))

def get_position_rep(position):
    board, move, winning = position
    return torch.tensor(get_board_rep(board)), torch.tensor(get_move_rep(move)), torch.tensor([winning]).float()

def get_move_rep(move: chess.Move):
    rep = np.zeros((76,8,8), dtype=np.float32)
    from_square, to_square = move.from_square, move.to_square
    x,y = to_coords(to_square)
    if move.promotion is not None:
        piece_type = move.promotion - 2
        from_x, from_y = to_coords(from_square)
        layer = 64 + piece_type * 3 + from_x - x + 1
        rep[layer,x,y] = 1
    else:
        rep[from_square,x,y] = 1
    return rep.reshape(4864)
    
def show_move_rep(rep):
    rep = torch.nn.functional.softmax(rep.cpu(), dim=1).numpy(force=True)
    vmin, vmax = np.min(rep), np.max(rep)
    rep = rep.reshape(76,8,8)
    fig,axs = plt.subplots(4,19,subplot_kw={"xticks":[], "yticks":[]},figsize=(20,5))
    for i in range(4):
        for j in range(19):
            layer = 19*i + j
            axs[i][j].imshow(rep[layer,:,:],vmin=vmin, vmax=vmax)
            if layer < 64:
                axs[i][j].set_title(f'{chess.square_name(layer)}')
            else:
                axs[i][j].set_title(f'{move_layer_names[layer-64]}')
    plt.show()

def sample_move(move_rep):
    move_rep = torch.nn.functional.softmax(move_rep.cpu()).numpy(force=True).reshape(4864)
    indices = np.arange(4864)
    index = np.random.choice(indices, p=move_rep)
    move = np.zeros((1,4864))
    move[0][index] = 1
    index = move.reshape(76,8,8).nonzero()
    from_square = index[1][0] // 19 + index[1][0]%19
    print(f"{chess.square_name(index[0][0])} to {from_coords(index[2][0], index[1][0])}")
    # layer = index // 19 + index % 19
    # print(layer)
    return move