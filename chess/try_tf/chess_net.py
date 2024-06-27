import numpy as np
import keras
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import chess
os.environ["ROCM_PATH"] = "/opt/rocm"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "-1"

try:
    if setup: # type: ignore
        print("Not resetting GPU")
except:
    setup = False
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(e)

    tf.config.experimental.set_virtual_device_configuration(
    gpus[0], 
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)]
    )

def get_board_rep(board: chess.Board):
    pieces = board.board_fen().split("/")
    print(pieces)
    board_rep = np.zeros((8,8,12))
    for (i,rank) in enumerate(pieces):
        j = 0
        while j < 8:
            piece = rank[j]
            if piece.isnumeric():
                j += int(piece)
            else:
                board_rep[i,j,piece_layer[piece]] = 1.0
                j+=1
    return board_rep
            

piece_layer = {"P":0,"R":1,"N":2,"B":3,"Q":4, "K":5,"p":6,"r":7,"n":8,"b":9,"q":10, "k":11}

board=chess.Board()
print(board.fen())
board_rep = get_board_rep(board)
fig,axs = plt.subplots(4,3)
for i in range(4):
    for j in range(3):
        axs[i][j].imshow(board_rep[:,:,3*i + j])