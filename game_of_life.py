import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
from matplotlib import colors
import numpy as np
torch.set_grad_enabled(False)

arg_parser = argparse.ArgumentParser(description='Simulate the  game of life.')



arg_parser.add_argument('--n_frames',
                       default=100,
                       type=int,
                       help='Number of frames')

arg_parser.add_argument("--board_width",
                       default=100,
                       type=int,
                       help='Board width (pixels/cells')

arg_parser.add_argument("--board_height",
                       default=100,
                       type=int,
                       help='Board width (pixels/cells')

arg_parser.add_argument("--seed",
                       default=0,
                       type=int,
                       help='Pytorch random seed')


arg_parser.add_argument("--scale_factor",
                       default=1,
                       type=int,
                       help='Scale up image (pixels per cell)')



args = arg_parser.parse_args()



torch.manual_seed(args.seed)

conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')
conv.weight = nn.Parameter(torch.tensor([[1,1,1],[1,0,1],[1,1,1]]).float().unsqueeze(0).unsqueeze(0))


board = torch.randint(0,2,(1,args.board_height,args.board_width)).float().unsqueeze(1)


boards = []


Path("boards/").mkdir(exist_ok=True)


print("Simulating")
for i in tqdm(range(0,args.n_frames)):
    boards.append(board.clone())
    neighbor_sums = conv(board)
    alive_mask = board==1
    dead_mask = ~alive_mask

    kill_mask = alive_mask & torch.logical_or( neighbor_sums<=1,neighbor_sums>3)

    give_life_mask = dead_mask & (neighbor_sums == 3)

    board[kill_mask] = 0
    board[give_life_mask] = 1
    
print(f"Saving images")


cmap = colors.ListedColormap(['black', 'white'])

for i, board in enumerate(tqdm(boards)):
    board = torch.nn.functional.interpolate(board,scale_factor  = args.scale_factor)
    
    plt.imsave(f"boards/{i}_board.png",board.squeeze(), cmap=cmap)










