"""
Minesweeper and Probabilistic AI Solver

Controls:
	Left click: open a cell
	Right click: flag a cell
	A: toggle showing AI solver probabilities
	Q: quick solve using AI (repeat until game over: flag all 100% prob cells; open the minimum-probability cell)
	Click/press any key to reset when game over

Author: Sam Barba
Created 28/01/2025
"""

import sys

import pygame as pg

from game import Game


if __name__ == '__main__':
	# game = Game(rows=9, cols=9, num_mines=10)  # Beginner (~12.3% mine density)
	# game = Game(rows=16, cols=16, num_mines=40)  # Intermediate (~15.6% mine density)
	# game = Game(rows=16, cols=30, num_mines=99)  # Expert (~20.6% mine density)
	game = Game(rows=25, cols=40, num_mines=150)  # Custom
	game.render()

	while True:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				sys.exit()

			elif event.type == pg.MOUSEBUTTONDOWN:
				if game.game_over:
					game.setup()
					game.render()
				elif event.button in (1, 3):
					x, y = event.pos
					is_left_click = event.button == 1
					game.handle_click(y, x, is_left_click=is_left_click, is_screen_coords=True)
					game.render()

			elif event.type == pg.KEYDOWN:
				if game.game_over:
					game.setup()
					game.render()
				elif event.key == pg.K_a:
					game.show_solver_probs = not game.show_solver_probs
					if game.show_solver_probs:
						game.solver.calculate_mine_probs()
					game.render()
				elif event.key == pg.K_q:
					game.auto_play()
