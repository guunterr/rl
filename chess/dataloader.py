import lichess.api
import matplotlib.pyplot as plt
import chess.pgn
import numpy as np
import itertools

path_to_games = "chess/data/lichess_2013/lichess_db_standard_rated_2013-06.pgn"
    
def position_generator(filepath):
    pgn = open(filepath)
    while True:
        game = chess.pgn.read_game(pgn)
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
        
def to_coords(square: chess.Square):
    return 7-chess.square_rank(square),chess.square_file(square)

def move_rep(move: chess.Move):
    rep = np.zeros((8,8,76))


    from_square, to_square = move.from_square, move.to_square
    x,y = to_coords(to_square)
    if move.promotion is not None:
        piece_type = move.promotion - 2
        from_x, from_y = to_coords(from_square)
        layer = 64 + piece_type * 3 + from_x - x + 1
        rep[x,y,layer] = 1
        print(x,y,piece_type, from_x, layer)
        return rep
    else:
        rep[x,y,from_square] = 1
        return rep
        
gen = position_generator(path_to_games)
pgn = open(path_to_games)
count = 0
for pos in gen:
    # print(move_rep(pos[1])[:,:,6])
    if pos[1].promotion is not None and pos[1].promotion != 5:
        print(count, pos[1], chess.piece_name(pos[1].promotion), chess.piece_symbol(pos[1].promotion), pos[1].promotion)
        for (i,layer) in enumerate(move_rep(pos[1])[:,:,64:]):
            if np.any(layer):
                print(i)
                print(layer)
        break
    count+=1
    if count > 100000:
        break
