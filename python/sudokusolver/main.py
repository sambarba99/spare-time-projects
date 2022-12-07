"""
Sudoku solver using depth-first search

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
FOREGROUND = (220, 220, 220)

# Puzzles in ascending order of difficulty
PRESET_PUZZLES = {'blank': '0' * 81,
	'easy': '000000000010020030000000000000000000040050060000000000000000000070080090000000000',
	'medium': '100000000020000000003000000000400000000050000000006000000000700000000080000000009',
	'hard': '120000034500000006000000000000070000000891000000020000000000000300000005670000089',
	'insane': '800000000003600000070090200050007000000045700000100030001000068008500010090000400'}

plt.rcParams['figure.figsize'] = (7, 5)

board = np.zeros((BOARD_SIZE, BOARD_SIZE)).astype(int)
given_ij = None  # Store coords of numbers that are already given
backtrack_grid = None  # For visualising backtracks
n_backtracks = 0

def solve():
	def is_full():
		return board.all()

	def find_free_square():
		for (i, j), val in np.ndenumerate(board):
			if val == 0:
				return i, j

		raise AssertionError("Shouldn't have got here")

	def legal(n, i, j):
		# Top-left coords of big square
		bi = i - (i % 3)
		bj = j - (j % 3)

		# Check row + column + big square
		return n not in board[i] \
			and n not in board[:, j] \
			and n not in board[bi:bi + 3, bj:bj + 3]

	if is_full(): return

	global n_backtracks

	for event in pg.event.get():
		if event.type == pg.QUIT:
			sys.exit()

	i, j = find_free_square()
	for n in range(1, 10):
		if legal(n, i, j):
			board[i][j] = n
			draw_grid(f'Solving ({n_backtracks} backtracks)')
			solve()

	if is_full(): return

	# If we're here, no numbers were legal
	# So the previous attempt in the loop must be invalid
	# So we reset the square in order to backtrack, so next number is tried
	board[i][j] = 0
	n_backtracks += 1
	backtrack_grid[i][j] += 1
	draw_grid(f'Solving ({n_backtracks} backtracks)')

def draw_grid(status):
	scene.fill((20, 20, 20))
	status_font = pg.font.SysFont('consolas', 14)
	cell_font = pg.font.SysFont('consolas', 24)

	status_lbl = status_font.render(status, True, FOREGROUND)
	scene.blit(status_lbl, (GRID_OFFSET, 32))

	for (i, j), val in np.ndenumerate(board):
		str_val = '' if val == 0 else str(val)

		if (i, j) in given_ij:
			# Draw already given numbers as green
			cell_lbl = cell_font.render(str_val, True, (0, 140, 0))
		else:
			cell_lbl = cell_font.render(str_val, True, FOREGROUND)

		lbl_rect = cell_lbl.get_rect(center=((j + 0.5) * CELL_SIZE + GRID_OFFSET,
			(i + 0.5) * CELL_SIZE + GRID_OFFSET + 1))
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
			match event.type:
				case pg.MOUSEBUTTONDOWN: return
				case pg.QUIT: sys.exit()

def plot_backtracks(difficulty_lvl):
	# Flip, as we want matplotlib to enumerate the y-axis from 0 to 8 going upwards
	# (line 'plt.gca().invert_yaxis()' below)
	grid_flipped = np.flipud(backtrack_grid)

	ax = plt.subplot()
	mat = ax.matshow(grid_flipped, cmap=plt.cm.plasma)
	ax.xaxis.set_ticks_position('bottom')
	for (j, i), val in np.ndenumerate(grid_flipped):
		ax.text(x=i, y=j, s=val, ha='center', va='center')
	ax.set_title(f'Heatmap of backtracks for difficulty level: {difficulty_lvl}')
	plt.gca().invert_yaxis()
	plt.colorbar(mat, ax=ax)
	plt.show()

if __name__ == '__main__':
	pg.init()
	pg.display.set_caption('Sudoku Solver')
	scene = pg.display.set_mode((BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET,
		BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET))

	while True:
		for difficulty_lvl, config in PRESET_PUZZLES.items():
			given_ij = []
			backtrack_grid = np.zeros((BOARD_SIZE, BOARD_SIZE)).astype(int)
			n_backtracks = 0

			for idx, n in enumerate(config):
				i, j = divmod(idx, BOARD_SIZE)
				board[i][j] = int(n)
				if n != '0': given_ij.append((i, j))

			draw_grid(f'Level: {difficulty_lvl} (click to solve)')
			wait_for_click()
			start = perf_counter()
			solve()
			interval = perf_counter() - start
			draw_grid(f'Solved ({n_backtracks} backtracks, {interval:.3}s) - click for next puzzle')
			plot_backtracks(difficulty_lvl)
			wait_for_click()
