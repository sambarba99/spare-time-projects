"""
N Queens solver using depth-first search

Author: Sam Barba
Created 08/02/2022
"""

import numpy as np
import pygame as pg
import sys

DRAW_STEPS = True
N = 12  # N > 3
BLANK = 0
QUEEN = 1
CELL_SIZE = 40
GRID_OFFSET = 60

board = np.zeros((N, N)).astype(int)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def solve(row=0):
	if row == N:  # If all queens placed
		return True

	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)

	for col in range(N):
		if valid(row, col):
			board[row][col] = QUEEN
			if DRAW_STEPS:
				draw_grid('Solving...')
			if solve(row + 1):
				return True

		# Reset the square in order to backtrack
		board[row][col] = BLANK
		if DRAW_STEPS:
			draw_grid('Solving...')

	return False

def valid(row, col):
	# Check row and column
	if QUEEN in board[row] or QUEEN in board[:, col]: return False

	# Check upper diagonal on left side
	for y, x in zip(range(row, -1, -1), range(col, -1, -1)):
		if board[y][x] == QUEEN:
			return False

	# Check upper diagonal on right side
	for y, x in zip(range(row, -1, -1), range(col, N)):
		if board[y][x] == QUEEN:
			return False

	return True

def draw_grid(solve_status):
	scene.fill((20, 20, 20))
	status_font = pg.font.SysFont('consolas', 16)
	cell_font = pg.font.SysFont('consolas', 24)

	status_lbl = status_font.render(f'{solve_status}', True, (220, 220, 220))
	scene.blit(status_lbl, (GRID_OFFSET, 25))

	for y in range(N):
		for x in range(N):
			if board[y][x] == QUEEN:
				cell_lbl = cell_font.render('Q', True, (220, 150, 0))
				lbl_rect = cell_lbl.get_rect(center=((x + 0.5) * CELL_SIZE + GRID_OFFSET,
					(y + 0.5) * CELL_SIZE + GRID_OFFSET))
				scene.blit(cell_lbl, lbl_rect)

	# Grid lines
	for i in range(GRID_OFFSET, N * CELL_SIZE + GRID_OFFSET + 1, CELL_SIZE):
		pg.draw.line(scene, (220, 220, 220), (i, GRID_OFFSET), (i, N * CELL_SIZE + GRID_OFFSET))
		pg.draw.line(scene, (220, 220, 220), (GRID_OFFSET, i), (N * CELL_SIZE + GRID_OFFSET, i))

	pg.display.update()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	pg.init()
	pg.display.set_caption('N Queens Solver')
	scene = pg.display.set_mode((N * CELL_SIZE + 2 * GRID_OFFSET, N * CELL_SIZE + 2 * GRID_OFFSET))

	solved = solve()
	draw_grid('Solved!' if solved else 'No solution')

	while True:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				sys.exit(0)
