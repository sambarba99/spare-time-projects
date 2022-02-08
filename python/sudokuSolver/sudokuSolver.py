# Sudoku solver using depth-first search
# Author: Sam Barba
# Created 07/09/2021

import pygame as pg
import sys

BOARD_SIZE = 9
CELL_SIZE = 50
GRID_OFFSET = 50
GRID_COLOUR = (220, 220, 220)

# Puzzles in ascending order of difficulty
PRESET_PUZZLES = {"Blank": "0" * 81,
	"Easy": "000000000010020030000000000000000000040050060000000000000000000070080090000000000",
	"Medium": "100000000020000000003000000000400000000050000000006000000000700000000080000000009",
	"Hard": "120000034500000006000000000000070000000891000000020000000000000300000005670000089",
	"Insane": "800000000003600000070090200050007000000045700000100030001000068008500010090000400"}

board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
givenYX = [] # Y before X, as 2D arrays are row-major
numBacktracks = 0

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def solve(scene, difficultyLvl):
	if isFull(): return

	global numBacktracks

	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)

	y, x = findFreeSquare()
	for n in range(1, 10):
		if legal(n, y, x):
			board[y][x] = n
			drawGrid(scene, difficultyLvl, "solving...")
			solve(scene, difficultyLvl)

	if isFull(): return

	# If we're here, no numbers were legal
	# So the previous attempt in the loop must be invalid
	# So we reset the square in order to backtrack, so next number is tried
	board[y][x] = 0
	numBacktracks += 1
	drawGrid(scene, difficultyLvl, "solving...")

def drawGrid(scene, difficultyLvl, solveStatus):
	scene.fill((20, 20, 20))
	statusFont = pg.font.SysFont("consolas", 16)
	cellFont = pg.font.SysFont("consolas", 30)

	statusLbl = statusFont.render(f"Difficulty: {difficultyLvl} ({solveStatus})", True, GRID_COLOUR)
	backtracksLbl = statusFont.render(f"{numBacktracks} backtracks", True, GRID_COLOUR)
	scene.blit(statusLbl, (GRID_OFFSET, 20))
	scene.blit(backtracksLbl, (GRID_OFFSET, 515))

	for y in range(BOARD_SIZE):
		for x in range(BOARD_SIZE):
			n = "" if board[y][x] == 0 else str(board[y][x])

			if [y, x] in givenYX:
				# Draw already given numbers as green
				cellLbl = cellFont.render(n, True, (0, 140, 0))
			else:
				cellLbl = cellFont.render(n, True, GRID_COLOUR)
			scene.blit(cellLbl, (x * CELL_SIZE + GRID_OFFSET + 17, y * CELL_SIZE + GRID_OFFSET + 12))

	# Thin grid lines
	for i in range(GRID_OFFSET, BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET, CELL_SIZE):
		pg.draw.line(scene, GRID_COLOUR, (i, GRID_OFFSET), (i, BOARD_SIZE * CELL_SIZE + GRID_OFFSET))
		pg.draw.line(scene, GRID_COLOUR, (GRID_OFFSET, i), (BOARD_SIZE * CELL_SIZE + GRID_OFFSET, i))

	# Thick grid lines
	for i in range(GRID_OFFSET + CELL_SIZE * 3, BOARD_SIZE * CELL_SIZE, CELL_SIZE * 3):
		pg.draw.line(scene, GRID_COLOUR, (i, GRID_OFFSET), (i, BOARD_SIZE * CELL_SIZE + GRID_OFFSET), 5)
		pg.draw.line(scene, GRID_COLOUR, (GRID_OFFSET, i), (BOARD_SIZE * CELL_SIZE + GRID_OFFSET, i), 5)

	pg.display.flip()

def isFull():
	return all(n != 0 for row in board for n in row)

def findFreeSquare():
	for y in range(BOARD_SIZE):
		for x in range(BOARD_SIZE):
			if board[y][x] == 0:
				return y, x

	raise AssertionError("Shouldn't have got here")

def legal(n, y, x):
	# Top-left coords of big square
	bigSquareY = y - (y % 3)
	bigSquareX = x - (x % 3)

	# Check big square
	for checkY in range(bigSquareY, bigSquareY + 3):
		for checkX in range(bigSquareX, bigSquareX + 3):
			if board[checkY][checkX] == n:
				return False

	# Check row and column
	if n in board[y] or n in [row[x] for row in board]:
		return False

	return True

def waitForClick():
	while True:
		for event in pg.event.get():
			if event.type == pg.MOUSEBUTTONDOWN:
				return
			elif event.type == pg.QUIT:
				pg.quit()
				sys.exit(0)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

pg.init()
pg.display.set_caption("Sudoku Solver")
scene = pg.display.set_mode((BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET, BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET))

# Game loop so pygame window doesn't close automatically
while True:
	for difficultyLvl, config in PRESET_PUZZLES.items():
		givenYX = []
		numBacktracks = 0

		for idx, n in enumerate(config):
			y, x = idx // BOARD_SIZE, idx % BOARD_SIZE
			board[y][x] = int(n)
			if int(n) != 0:
				givenYX.append([y, x])

		drawGrid(scene, difficultyLvl, "click to solve")
		waitForClick()
		solve(scene, difficultyLvl)
		drawGrid(scene, difficultyLvl, "solved! Click for next puzzle")
		waitForClick()
