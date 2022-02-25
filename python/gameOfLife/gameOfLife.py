# Game of Life
# Author: Sam Barba
# Created 21/09/2021

# 1: Preset pattern 1 (glider gun)
# 2: Preset pattern 2 (R pentomino)
# R: Reset and randomise
# Space: Play/pause

import pygame as pg
import random
import sys

ROWS = 140
COLS = 240
CELL_SIZE = 7

GLIDER_GUN = [(65, 105), (65, 106), (66, 105), (66, 106), (63, 117), (63, 118), (64, 116), (64, 120),
	(65, 115), (65, 121), (66, 115), (66, 119), (66, 121), (66, 122), (67, 115), (67, 121), (68, 116),
	(68, 120), (69, 117), (69, 118), (63, 125), (63, 126), (64, 125), (64, 126), (65, 125), (65, 126),
	(62, 127), (66, 127), (61, 129), (62, 129), (66, 129), (67, 129), (63, 139), (63, 140), (64, 139),
	(64, 140)]

R_PENTOMINO = [(68, 119), (68, 120), (69, 118), (69, 119), (70, 119)]

grid = None
running = True

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def update_grid():
	global grid  # 2D array of Booleans (true = alive)
	next_gen_grid = [[False] * COLS for _ in range(ROWS)]

	for y in range(ROWS):
		for x in range(COLS):
			n = count_live_neighbours(y, x)

			if n < 2 or n > 3:
				next_gen_grid[y][x] = False
			elif n == 3:
				next_gen_grid[y][x] = True
			else:
				next_gen_grid[y][x] = grid[y][x]

	grid = next_gen_grid[:]

def count_live_neighbours(y, x):
	n = 0
	for y_offset in range(-1, 2):
		for x_offset in range(-1, 2):
			check_y = y + y_offset
			check_x = x + x_offset

			if check_y in range(ROWS) and check_x in range(COLS) \
				and grid[check_y][check_x]:
				n += 1

	return n - 1 if grid[y][x] else n

def set_pattern(pattern):
	global grid
	grid = [[(y, x) in pattern for x in range(COLS)] for y in range(ROWS)]

def randomise_live_cells():
	global grid
	grid = [[random.random() < 0.2 for _ in range(COLS)] for _ in range(ROWS)]

def draw_grid():
	scene.fill((50, 50, 50))

	for y in range(ROWS):
		for x in range(COLS):
			if grid[y][x]:
				pg.draw.rect(scene, (220, 140, 0), pg.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

	pg.display.update()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

randomise_live_cells()

pg.init()
pg.display.set_caption("Game of Life")
scene = pg.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE))

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)
		elif event.type == pg.KEYDOWN:
			if event.key == pg.K_1:  # Preset 1
				set_pattern(GLIDER_GUN)
			elif event.key == pg.K_2:  # Preset 2
				set_pattern(R_PENTOMINO)
			elif event.key == pg.K_r:  # Reset and randomise
				randomise_live_cells()
			elif event.key == pg.K_SPACE:  # Play/pause
				running = not running

	draw_grid()
	if running: update_grid()
