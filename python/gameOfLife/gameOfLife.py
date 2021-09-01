# Game of Life
# Author: Sam Barba
# Created 21/09/2021

# 1 - 3: presets
# C: clear and reset
# R: randomise live cells
# Space: play/pause

import pygame as pg
import random
import sys

OSCILLATORS = [[24, 10], [35, 10], [36, 10], [24, 11], [29, 11], [30, 11], [31, 11], [35, 11], [24, 12],
	[28, 12], [29, 12], [30, 12], [38, 12], [37, 13], [38, 13], [27, 17], [28, 17], [29, 17], [33, 17],
	[34, 17], [35, 17], [25, 19], [30, 19], [32, 19], [37, 19], [25, 20], [30, 20], [32, 20], [37, 20],
	[25, 21], [30, 21], [32, 21], [37, 21], [27, 22], [28, 22], [29, 22], [33, 22], [34, 22], [35, 22],
	[27, 24], [28, 24], [29, 24], [33, 24], [34, 24], [35, 24], [25, 25], [30, 25], [32, 25], [37, 25],
	[25, 26], [30, 26], [32, 26], [37, 26], [25, 27], [30, 27], [32, 27], [37, 27], [27, 29], [28, 29],
	[29, 29], [33, 29], [34, 29], [35, 29], [46, 14], [46, 15], [45, 16], [47, 16], [46, 17], [46, 18],
	[46, 19], [46, 20], [45, 21], [47, 21], [46, 22], [46, 23]]

GLIDER_GUN = [[11, 15], [12, 15], [11, 16], [12, 16], [23, 13], [24, 13], [22, 14], [26, 14], [21, 15],
	[27, 15], [21, 16], [25, 16], [27, 16], [28, 16], [21, 17], [27, 17], [22, 18], [26, 18], [23, 19],
	[24, 19], [31, 13], [32, 13], [31, 14], [32, 14], [31, 15], [32, 15], [33, 12], [33, 16], [35, 11],
	[35, 12], [35, 16], [35, 17], [45, 13], [46, 13], [45, 14], [46, 14]]

R_PENTOMINO = [[119, 68], [120, 68], [118, 69], [119, 69], [119, 70]]

FPS = 6

rows = 41
cols = 70
cellSize = 22
running = True

# ---------------------------------------------------------------------------------------------------- #
# ---------------------------------------------  CLASSES  -------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

class Cell:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.isAlive = False

	def countLiveNeighbours(self, grid):
		n = 0
		for xOffset in range(-1, 2):
			for yOffset in range(-1, 2):
				checkX = self.x + xOffset
				checkY = self.y + yOffset
				if 0 <= checkX < len(grid) and 0 <= checkY < len(grid[0]) and grid[checkX][checkY].isAlive:
					n += 1

		return n - 1 if self.isAlive else n

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def draw(grid, scene):
	for x in range(cols):
		for y in range(rows):
			c = (255, 160, 0) if grid[x][y].isAlive else (80, 80, 80)
			pg.draw.rect(scene, c, pg.Rect(x * cellSize, y * cellSize, cellSize, cellSize))

	pg.display.flip()

def update(grid):
	nextGenGrid = [[Cell(x, y) for y in range(rows)] for x in range(cols)]

	for x in range(cols):
		for y in range(rows):
			n = grid[x][y].countLiveNeighbours(grid)

			if n < 2 or n > 3: nextGenGrid[x][y].isAlive = False
			elif n == 3: nextGenGrid[x][y].isAlive = True
			else: nextGenGrid[x][y].isAlive = grid[x][y].isAlive

	for x in range(cols):
		for y in range(rows):
			grid[x][y].isAlive = nextGenGrid[x][y].isAlive

def setPattern(grid, pattern):
	for x, y in pattern:
		grid[x][y].isAlive = True

def randomiseLiveCells(grid):
	allCoords = [(x, y) for x in range(cols) for y in range(rows)]
	liveCellCoords = random.sample(allCoords, round(cols * rows * 0.1))

	for x, y in liveCellCoords:
		grid[x][y].isAlive = True

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Usually other way around; this is for the sake of using x,y rather than y,x
grid = [[Cell(x, y) for y in range(rows)] for x in range(cols)]
randomiseLiveCells(grid)

pg.init()
pg.display.set_caption("Game of Life")
scene = pg.display.set_mode((cols * cellSize, rows * cellSize))
clock = pg.time.Clock()

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)
		elif event.type == pg.KEYDOWN:
			if event.key == pg.K_1: # Preset 1
				rows, cols, cellSize = 41, 70, 22
				grid = [[Cell(x, y) for y in range(rows)] for x in range(cols)]
				setPattern(grid, OSCILLATORS)

			elif event.key == pg.K_2: # Preset 2
				rows, cols, cellSize = 41, 70, 22
				grid = [[Cell(x, y) for y in range(rows)] for x in range(cols)]
				setPattern(grid, GLIDER_GUN)

			elif event.key == pg.K_3: # Preset 3
				rows, cols, cellSize = 140, 240, 6
				grid = [[Cell(x, y) for y in range(rows)] for x in range(cols)]
				setPattern(grid, R_PENTOMINO)

			elif event.key in [pg.K_c, pg.K_r]: # Clear and reset / randomise
				rows, cols, cellSize = 41, 70, 22
				grid = [[Cell(x, y) for y in range(rows)] for x in range(cols)]
				randomiseLiveCells(grid)

			elif event.key == pg.K_SPACE: # Play/pause
				running = not running

	draw(grid, scene)
	if running: update(grid)

	clock.tick(FPS)
