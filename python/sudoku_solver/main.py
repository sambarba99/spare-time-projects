"""
Sudoku solver using Depth-First Search and backtracking

Author: Sam Barba
Created 07/09/2021
"""

import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg


BOARD_SIZE = 9
CELL_SIZE = 50
GRID_OFFSET = 75
FOREGROUND = (224, 224, 224)

# Puzzles in ascending order of difficulty
PRESET_PUZZLES = {
	'blank': '0' * 81,
	'easy': '000000000010020030000000000000000000040050060000000000000000000070080090000000000',
	'medium': '100000000020000000003000000000400000000050000000006000000000700000000080000000009',
	'hard': '120000034500000006000000000000070000000891000000020000000000000300000005670000089',
	'world hardest': '800000000003600000070090200050007000000045700000100030001000068008500010090000400'
}

board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
given_coords = None  # Store coords of numbers that are already given
backtrack_grid = None  # For visualising backtracks
num_backtracks = 0


def solve():
	"""Non-recursive depth-first search with backtracking to solve the board"""

	def find_free_square():
		return next(((y, x) for (y, x), val in np.ndenumerate(board) if val == 0), None)

	def is_legal(n, y, x):
		# Top-left coords of big square
		by = y - (y % 3)
		bx = x - (x % 3)

		# Check row + column + big square
		return n not in board[y] and n not in board[:, x] and n not in board[by:by + 3, bx:bx + 3]


	global num_backtracks

	y, x = find_free_square()
	stack = [(1, y, x)]  # Start checking numbers from 1

	while stack:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				sys.exit()

		n, y, x = stack.pop()

		if n > 9:
			# No possible number found for (y, x), backtrack
			board[y, x] = 0
			num_backtracks += 1
			backtrack_grid[y, x] += 1
			draw_grid(f'Solving ({num_backtracks} backtracks)')
		elif is_legal(n, y, x):
			board[y, x] = n
			draw_grid(f'Solving ({num_backtracks} backtracks)')
			next_free = find_free_square()
			if next_free:
				# Append (n + 1, y, x) even though we've just placed n at (y,x) to prepare for possible backtracking
				stack.append((n + 1, y, x))
				stack.append((1, *next_free))
			else:
				return  # Solved
		else:
			stack.append((n + 1, y, x))  # Try the next number


def draw_grid(status):
	scene.fill((16, 16, 16))

	status_lbl = status_font.render(status, True, FOREGROUND)
	scene.blit(status_lbl, (GRID_OFFSET, 32))

	for (y, x), val in np.ndenumerate(board):
		str_val = '' if val == 0 else str(val)

		if (y, x) in given_coords:
			# Draw already given numbers as green
			cell_lbl = cell_font.render(str_val, True, (0, 150, 0))
		else:
			cell_lbl = cell_font.render(str_val, True, FOREGROUND)

		lbl_rect = cell_lbl.get_rect(
			center=((x + 0.5) * CELL_SIZE + GRID_OFFSET, (y + 0.5) * CELL_SIZE + GRID_OFFSET + 1)
		)
		scene.blit(cell_lbl, lbl_rect)

	# Thin grid lines
	for y in range(GRID_OFFSET, BOARD_SIZE * CELL_SIZE + GRID_OFFSET + 1, CELL_SIZE):
		pg.draw.line(scene, FOREGROUND, (y, GRID_OFFSET), (y, BOARD_SIZE * CELL_SIZE + GRID_OFFSET))
		pg.draw.line(scene, FOREGROUND, (GRID_OFFSET, y), (BOARD_SIZE * CELL_SIZE + GRID_OFFSET, y))

	# Thick grid lines
	for y in range(GRID_OFFSET + CELL_SIZE * 3, BOARD_SIZE * CELL_SIZE, CELL_SIZE * 3):
		pg.draw.line(scene, FOREGROUND, (y, GRID_OFFSET), (y, BOARD_SIZE * CELL_SIZE + GRID_OFFSET), 5)
		pg.draw.line(scene, FOREGROUND, (GRID_OFFSET, y), (BOARD_SIZE * CELL_SIZE + GRID_OFFSET, y), 5)

	pg.display.update()


def wait_for_click():
	while True:
		for event in pg.event.get():
			match event.type:
				case pg.MOUSEBUTTONDOWN:
					return
				case pg.QUIT:
					sys.exit()


def plot_backtracks(difficulty_lvl):
	threshold = (backtrack_grid.min() + backtrack_grid.max()) / 2
	_, ax = plt.subplots(figsize=(7, 5))
	mat = ax.matshow(backtrack_grid, cmap='Blues')
	for (j, i), val in np.ndenumerate(backtrack_grid):
		ax.text(x=i, y=j, s=val, ha='center', va='center', color='black' if val < threshold else 'white')
	ax.set_title(f'Heatmap of backtracks for difficulty level: {difficulty_lvl}')
	plt.axis('off')
	plt.colorbar(mat, ax=ax)
	plt.show()


if __name__ == '__main__':
	pg.init()
	pg.display.set_caption('Sudoku Solver')
	scene = pg.display.set_mode((BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET,
		BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET))
	status_font = pg.font.SysFont('consolas', 14)
	cell_font = pg.font.SysFont('consolas', 24)

	while True:
		for difficulty_lvl, config in PRESET_PUZZLES.items():
			given_coords = []
			backtrack_grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
			num_backtracks = 0

			for idx, n in enumerate(config):
				y, x = divmod(idx, BOARD_SIZE)
				board[y, x] = int(n)
				if n != '0': given_coords.append((y, x))

			draw_grid(f'Level: {difficulty_lvl} (click to solve)')
			wait_for_click()
			start = perf_counter()
			solve()
			interval = perf_counter() - start
			draw_grid(f'Solved ({num_backtracks} backtracks, {interval:.3}s) - click for next puzzle')
			plot_backtracks(difficulty_lvl)
			wait_for_click()
