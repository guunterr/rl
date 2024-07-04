import os
os.environ["ROCM_PATH"] = "/opt/rocm"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import chess_net, utils, torch, numpy as np, parse_data, chess

def get_move(board: chess.Board):
    while True:
        move = input("Input move in UCI: ")
        try:
            move = chess.Move.from_uci(move)
            if board.is_legal(move):
                return move
            print("Invalid move!")
        except:
            print("Invalid move!")
            pass

if __name__ == "__main__":
    model = chess_net.ChessModel().to('cuda')
    state_dict = torch.load("/home/gerard/Documents/Personal/Programming/rl/chessai/checkpoints/1/checkpoint_0_41600.pt")['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    board = chess.Board()
    white = True
    player = input("(b) for model plays black (w) for model plays white")
    if player == "b":
        white = False
        print("ENTER PLAYER MOVE: ")
        board.push(get_move(board))
    while True:
        print(board)
        board_rep = utils.get_board_rep(board)
        board_rep_tensor = torch.tensor(board_rep).to('cuda', torch.float32)
        p, v = model(board_rep_tensor.unsqueeze(0))
        # utils.show_move_rep(p)
        while True:
            rep, move = utils.sample_move(p)
            if move in board.legal_moves:
                break
        rep = torch.tensor(rep)
        # utils.show_move_rep(rep.reshape(76,8,8), False)
        # print("ENTER AI MOVE: ")
        board.push(move)
        print(board)
        print("ENTER PLAYER MOVE: ")
        board.push(get_move(board))
