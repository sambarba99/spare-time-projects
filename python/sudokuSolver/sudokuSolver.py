"""
Sudoku solver using depth-first search

Author: Sam Barba
Created 07/09/2021
"""

import numpy as np
import pygame as pg
import sys

BOARD_SIZE = 9
CELL_SIZE = 50
GRID_OFFSET = 75
FOREGROUND = (220, 220, 220)

# Puzzles in ascending order of difficulty
PRESET_PUZZLES = {'Blank': '0' * 81,
	'Easy': '000000000010020030000000000000000000040050060000000000000000000070080090000000000',
	'Medium': '100000000020000000003000000000400000000050000000006000000000700000000080000000009',
	'Hard': '120000034500000006000000000000070000000891000000020000000000000300000005670000089',
	'Insane': '800000000003600000070090200050007000000045700000100030001000068008500010090000400'}

board = np.zeros((BOARD_SIZE, BOARD_SIZE)).astype(int)
given_yx = []  # Y before X, as 2D arrays are row-major
num_backtracks = 0

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def solve(difficulty_lvl):
	if is_full(): return

	global num_backtracks

	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)

	y, x = find_free_square()
	for n in range(1, 10):
		if legal(n, y, x):
			board[y][x] = n
			draw_grid(difficulty_lvl=difficulty_lvl, solve_status='solving...')
			solve(difficulty_lvl)

	if is_full(): return

	# If we're here, no numbers were legal
	# So the previous attempt in the loop must be invalid
	# So we reset the square in order to backtrack, so next number is tried
	board[y][x] = 0
	num_backtracks += 1
	draw_grid(difficulty_lvl=difficulty_lvl, solve_status='solving...')

def is_full():
	return np.all(board)

def find_free_square():
	for (y, x), val in np.ndenumerate(board):
		if val == 0:
			return y, x

	raise AssertionError("Shouldn't have got here")

def legal(n, y, x):
	# Top-left coords of big square
	by = y - (y % 3)
	bx = x - (x % 3)

	# Check row + column + big square
	return n not in board[y] \
		and n not in board[:, x] \
		and n not in board[by:by + 3, bx:bx + 3]

def draw_grid(*, difficulty_lvl, solve_status):
	scene.fill((20, 20, 20))
	status_font = pg.font.SysFont('consolas', 16)
	cell_font = pg.font.SysFont('consolas', 30)

	status_lbl = status_font.render(f'Difficulty: {difficulty_lvl} ({solve_status})', True, FOREGROUND)
	backtracks_lbl = status_font.render(f'{num_backtracks} backtracks', True, FOREGROUND)
	scene.blit(status_lbl, (GRID_OFFSET, 32))
	scene.blit(backtracks_lbl, (GRID_OFFSET, 550))

	for (y, x), val in np.ndenumerate(board):
		str_val = '' if val == 0 else str(val)

		if (y, x) in given_yx:
			# Draw already given numbers as green
			cell_lbl = cell_font.render(str_val, True, (0, 140, 0))
		else:
			cell_lbl = cell_font.render(str_val, True, FOREGROUND)

		lbl_rect = cell_lbl.get_rect(center=((x + 0.5) * CELL_SIZE + GRID_OFFSET,
			(y + 0.5) * CELL_SIZE + GRID_OFFSET + 1))
		scene.blit(cell_lbl, lbl_rect)

	# Thin grid lines
	for i in range(GRID_OFFSET, BOARD_SIZE * CELL_SIZE + GRID_OFFSET + 1, CELL_SIZE):
		pg.draw.line(scene, FOREGROUND, (i, GRID_OFFSET), (i, BOARD_SIZE * CELL_SIZE + GRID_OFFSET))
		pg.draw.line(scene, FOREGROUND, (GRID_OFFSET, i), (BOARD_SIZE * CELL_SIZE + GRID_OFFSET, i))

	# Thick grid lines
	for i in range(GRID_OFFSET + CELL_SIZE * 3, BOARD_SIZE * CELL_SIZE, CELL_SIZE * 3):
		pg.draw.line(scene, FOREGROUND, (i, GRID_OFFSET), (i, BOARD_SIZE * CELL_SIZE + GRID_OFFSET), 5)
		pg.draw.line(scene, FOREGROUND, (GRID_OFFSET, i), (BOARD_SIZE * CELL_SIZE + GRID_OFFSET, i), 5)

	pg.display.update()

def wait_for_click():
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

if __name__ == '__main__':
	pg.init()
	pg.display.set_caption('Sudoku Solver')
	scene = pg.display.set_mode((BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET,
		BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET))

	while True:
		for difficulty_lvl, config in PRESET_PUZZLES.items():
			for idx, n in enumerate(config):
				y, x = idx // BOARD_SIZE, idx % BOARD_SIZE
				board[y][x] = int(n)

			num_backtracks = 0
			given_yx = [(y, x) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if board[y][x] != 0]

			draw_grid(difficulty_lvl=difficulty_lvl, solve_status='click to solve')
			wait_for_click()
			solve(difficulty_lvl)
			draw_grid(difficulty_lvl=difficulty_lvl, solve_status='solved! Click for next puzzle')
			wait_for_click()
