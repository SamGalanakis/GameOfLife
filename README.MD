
# Simple game of life simulator using pytorch convolutions

[Game of life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)

[Example_1](https://youtu.be/o1gV6niI8kY)  [Example_2](https://youtu.be/3t8ngsmviu8)

 
## Start simulation and save imags to boards/


```
python game_of_life.py --board_width 192 --board_height 108 --scale_factor 10 --n_frames 1000 --seed 0
```

Board height and width refers to the number of cells. Simulation is run at that resolution and then upscaled by a multiple according to scale factor. So scale_factor = 1 -> 1 pixel per cell.
Seed corresponds to the pytorch seed used to generate the initial board state.

## Output video from saved board images
```
ffmpeg -framerate 12 -i "boards/%d_board.png" video.mp4
```