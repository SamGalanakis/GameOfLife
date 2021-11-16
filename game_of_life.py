import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from pathlib import Path
torch.set_grad_enabled(False)
import sys
from tqdm import tqdm

def visualize_board(board):
    if len(board.shape)>3:
        board = board.squeeze()
    return plt.imshow(board)




n_iters = 10000
board_size = 1000



conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')
conv.weight = nn.Parameter(torch.tensor([[1,1,1],[1,0,1],[1,1,1]]).float().unsqueeze(0).unsqueeze(0))


board = torch.randint(0,2,(1,board_size,board_size)).float().unsqueeze(1)


boards = []


Path("boards/").mkdir(exist_ok=True)

for i in tqdm(range(0,n_iters)):
    boards.append(board.clone())
    neighbor_sums = conv(board)
    alive_mask = board==1
    dead_mask = ~alive_mask

    kill_mask = alive_mask & torch.logical_or( neighbor_sums<=1,neighbor_sums>3)

    give_life_mask = dead_mask & (neighbor_sums == 3)

    board[kill_mask] = 0
    board[give_life_mask] = 1
    

for i, board in enumerate(tqdm(boards)):
    plt.imsave(f"boards/{i}_board.png",board.squeeze())
# def animate(i):
#     neighbor_sums = conv(board)
#     alive_mask = board==1
#     dead_mask = ~alive_mask

#     kill_mask = alive_mask & torch.logical_or( neighbor_sums<=1,neighbor_sums>3)

#     give_life_mask = dead_mask & (neighbor_sums == 3)

#     board[kill_mask] = 0
#     board[give_life_mask] = 1

#     im = visualize_board(board)
#     return im

# fig,ax = plt.subplots(figsize=(9, 4.5))
# plt.xlim(0, board_size)
# plt.ylim(board_size, 0)
# ani = FuncAnimation(fig, animate, frames=n_iters, repeat=False, interval=10)

# #plt.show()
# writervideo = animation.FFMpegWriter(fps=60) 

# path = Path("animation.mp4")
# ani.save(path, writer=writervideo)







