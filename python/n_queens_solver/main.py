"""
N Queens solver using depth-first search

Author: Sam Barba
Created 08/02/2022
"""

import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg


N = 12
BLANK = 0
QUEEN = 1
CELL_SIZE = 30
GRID_OFFSET = 45

board = np.zeros((N, N), dtype=int)
backtrack_grid = np.zeros((N, N), dtype=int)  # For visualising backtracks


def solve():
	"""Non-recursive depth-first search with backtracking to solve the board"""

	stack = []
	row = col = 0  # Start from 0,0

	while True:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				sys.exit()

		if row == N:
			return True  # All queens placed

		placed = False
		while col < N:
			if valid(row, col):
				board[row, col] = QUEEN
				stack.append((row, col))  # Save state to stack in case of possible backtracking
				draw_grid('Solving...')

				# Move to next row
				row += 1
				col = 0
				placed = True
				break
			else:
				col += 1

		# If no column was valid, backtrack
		if not placed:
			if not stack:
				return False  # No more states to backtrack to means no solution

			row, last_col = stack.pop()
			board[row, last_col] = BLANK
			backtrack_grid[row, last_col] += 1
			draw_grid('Solving...')

			# Try the next column in the same row
			col = last_col + 1


def valid(row, col):
	# Check row and column
	if QUEEN in board[row] or QUEEN in board[:, col]: return False

	# Check upper diagonal on left side
	for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
		if board[i, j] == QUEEN:
			return False

	# Check upper diagonal on right side
	for i, j in zip(range(row, -1, -1), range(col, N)):
		if board[i, j] == QUEEN:
			return False

	return True


def draw_grid(solve_status):
	scene.fill('black')

	status_lbl = status_font.render(f'{solve_status}', True, (224, 224, 224))
	scene.blit(status_lbl, (GRID_OFFSET, 20))

	for i in range(N):
		for j in range(N):
			if (i + j) % 2:
				pg.draw.rect(
					scene, (16, 16, 16),
					pg.Rect(j * CELL_SIZE + GRID_OFFSET, i * CELL_SIZE + GRID_OFFSET, CELL_SIZE, CELL_SIZE)
				)
			else:
				pg.draw.rect(
					scene, (64, 64, 64),
					pg.Rect(j * CELL_SIZE + GRID_OFFSET, i * CELL_SIZE + GRID_OFFSET, CELL_SIZE, CELL_SIZE)
				)

			if board[i, j] == QUEEN:
				cell_lbl = cell_font.render('Q', True, (220, 150, 0))
				lbl_rect = cell_lbl.get_rect(
					center=((j + 0.5) * CELL_SIZE + GRID_OFFSET, (i + 0.5) * CELL_SIZE + GRID_OFFSET)
				)
				scene.blit(cell_lbl, lbl_rect)

	pg.display.update()


def plot_backtracks():
	threshold = (backtrack_grid.min() + backtrack_grid.max()) / 2
	_, ax = plt.subplots(figsize=(7, 5))
	mat = ax.matshow(backtrack_grid, cmap='Blues')
	for (j, i), val in np.ndenumerate(backtrack_grid):
		ax.text(x=i, y=j, s=val, ha='center', va='center', color='black' if val < threshold else 'white')
	ax.set_title(f'Heatmap of backtracks (total: {backtrack_grid.sum()})')
	plt.axis('off')
	plt.colorbar(mat, ax=ax)
	plt.show()


if __name__ == '__main__':
	pg.init()
	pg.display.set_caption('N Queens Solver')
	scene = pg.display.set_mode((N * CELL_SIZE + 2 * GRID_OFFSET, N * CELL_SIZE + 2 * GRID_OFFSET))
	status_font = pg.font.SysFont('consolas', 16)
	cell_font = pg.font.SysFont('consolas', 24)

	start = perf_counter()
	solved = solve()
	interval = perf_counter() - start

	draw_grid(f'Solved in {interval:.1f}s' if solved else 'No solution')
	plot_backtracks()

	while True:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				sys.exit()
