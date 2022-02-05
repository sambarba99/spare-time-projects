# Game of Life
# Author: Sam Barba
# Created 21/09/2021

# 1 - 3: Presets
# R: Reset and randomise
# Space: Play/pause

import pygame as pg
import random
import sys

OSCILLATORS = [[10, 24], [10, 35], [10, 36], [11, 24], [11, 29], [11, 30], [11, 31], [11, 35], [12, 24],
	[12, 28], [12, 29], [12, 30], [12, 38], [13, 37], [13, 38], [17, 27], [17, 28], [17, 29], [17, 33],
	[17, 34], [17, 35], [19, 25], [19, 30], [19, 32], [19, 37], [20, 25], [20, 30], [20, 32], [20, 37],
	[21, 25], [21, 30], [21, 32], [21, 37], [22, 27], [22, 28], [22, 29], [22, 33], [22, 34], [22, 35],
	[24, 27], [24, 28], [24, 29], [24, 33], [24, 34], [24, 35], [25, 25], [25, 30], [25, 32], [25, 37],
	[26, 25], [26, 30], [26, 32], [26, 37], [27, 25], [27, 30], [27, 32], [27, 37], [29, 27], [29, 28],
	[29, 29], [29, 33], [29, 34], [29, 35], [14, 46], [15, 46], [16, 45], [16, 47], [17, 46], [18, 46],
	[19, 46], [20, 46], [21, 45], [21, 47], [22, 46], [23, 46]]

GLIDER_GUN = [[15, 11], [15, 12], [16, 11], [16, 12], [13, 23], [13, 24], [14, 22], [14, 26], [15, 21],
	[15, 27], [16, 21], [16, 25], [16, 27], [16, 28], [17, 21], [17, 27], [18, 22], [18, 26], [19, 23],
	[19, 24], [13, 31], [13, 32], [14, 31], [14, 32], [15, 31], [15, 32], [12, 33], [16, 33], [11, 35],
	[12, 35], [16, 35], [17, 35], [13, 45], [13, 46], [14, 45], [14, 46]]

R_PENTOMINO = [[68, 119], [68, 120], [69, 118], [69, 119], [70, 119]]

FPS = 6

rows = 41
cols = 70
cellSize = 22
running = True

# ---------------------------------------------------------------------------------------------------- #
# ---------------------------------------------  CLASSES  -------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

class Cell:
	def __init__(self, y, x): # Y before X, as 2D arrays are row-major
		self.y = y
		self.x = x
		self.isAlive = False

	def countLiveNeighbours(self, grid):
		n = 0
		for yOffset in range(-1, 2):
			for xOffset in range(-1, 2):
				checkY = self.y + yOffset
				checkX = self.x + xOffset
				if 0 <= checkY < len(grid) and 0 <= checkX < len(grid[0]) and grid[checkY][checkX].isAlive:
					n += 1

		return n - 1 if self.isAlive else n

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def draw(grid, scene):
	scene.fill((80, 80, 80))

	for y in range(rows):
		for x in range(cols):
			if grid[y][x].isAlive:
				pg.draw.rect(scene, (255, 160, 0), pg.Rect(x * cellSize, y * cellSize, cellSize, cellSize))

	pg.display.flip()

def update(grid):
	nextGenGrid = [[Cell(y, x) for x in range(cols)] for y in range(rows)]

	for y in range(rows):
		for x in range(cols):
			n = grid[y][x].countLiveNeighbours(grid)

			if n < 2 or n > 3: nextGenGrid[y][x].isAlive = False
			elif n == 3: nextGenGrid[y][x].isAlive = True
			else: nextGenGrid[y][x].isAlive = grid[y][x].isAlive

	for y in range(rows):
		for x in range(cols):
			grid[y][x].isAlive = nextGenGrid[y][x].isAlive

def setPattern(grid, pattern):
	for y, x in pattern:
		grid[y][x].isAlive = True

def randomiseLiveCells(grid):
	allCoords = [(y, x) for x in range(cols) for y in range(rows)]
	liveCellCoords = random.sample(allCoords, round(cols * rows * 0.1))

	for y, x in liveCellCoords:
		grid[y][x].isAlive = True

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

grid = [[Cell(y, x) for x in range(cols)] for y in range(rows)]
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
			if event.key in [pg.K_1, pg.K_2, pg.K_3, pg.K_r]:
				rows, cols, cellSize = 41, 70, 22

				if event.key == pg.K_3:
					rows, cols, cellSize = 140, 240, 6

				grid = [[Cell(y, x) for x in range(cols)] for y in range(rows)]

			if event.key == pg.K_1: # Preset 1
				setPattern(grid, OSCILLATORS)
			elif event.key == pg.K_2: # Preset 2
				setPattern(grid, GLIDER_GUN)
			elif event.key == pg.K_3: # Preset 3
				setPattern(grid, R_PENTOMINO)
			elif event.key == pg.K_r: # Reset and randomise
				randomiseLiveCells(grid)
			elif event.key == pg.K_SPACE: # Play/pause
				running = not running

	draw(grid, scene)
	if running: update(grid)

	clock.tick(FPS)
