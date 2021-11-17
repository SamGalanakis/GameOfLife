
# Simple game of life simulator using pytorch convolutions

[Game of life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)

[Example_1](https://youtu.be/UOxtpOc5N9o) [Example_2](https://youtu.be/o1gV6niI8kY) 

## Start simulation and save images to boards/

```
python game_of_life.py --board_width 192 --board_height 108 --scale_factor 10 --n_frames 1000 --seed 0
```

Board height and width refers to the number of cells. Simulation is run at that resolution and then upscaled by a multiple according to scale factor. So scale_factor = 1 -> 1 pixel per cell.
Seed corresponds to the pytorch seed used to generate the initial board state. 

## Output video from saved board images

[ffmpeg](https://www.ffmpeg.org/) must be installed and on the path.

```
ffmpeg -framerate 12 -i "boards/%d_board.png" video.mp4
```
