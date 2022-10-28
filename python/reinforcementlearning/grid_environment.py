"""
Environment class for reinforcement learning algorithms demo

Author: Sam Barba
Created 25/02/2022
"""

import numpy as np
import pygame as pg

# Possible actions
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3

# Rendering constants
CELL_SIZE = 100
START_COL = (0, 140, 0)
GOLD_COL = (220, 120, 0)
HOLE_COL = (170, 0, 0)
TEXT_COL = (220, 220, 220)
ARROW_COL = (0, 80, 180)
NORTH_ARROW = ((0, 40), (15, 0), (30, 40))
SOUTH_ARROW = ((0, 0), (15, 40), (30, 0))
EAST_ARROW = ((0, 0), (0, 30), (40, 15))
WEST_ARROW = ((0, 15), (40, 0), (40, 30))

class GridEnv:
	def __init__(self, size=6):
		self.size = size
		self.actions = [NORTH, EAST, SOUTH, WEST]
		self.start = None
		self.gold = None
		self.hole = None
		self.rewards = None
		self.generate()

	def step(self, state, action):
		new_y, new_x = state
		if action == NORTH: new_y -= 1
		if action == EAST: new_x += 1
		if action == SOUTH: new_y += 1
		if action == WEST: new_x -= 1

		# If new state is within grid, use new coords
		if new_y in range(self.size) and new_x in range(self.size):
			state = (new_y, new_x)

		reward = self.rewards.get(state, 0) - 1  # -1 to penalise each time step
		terminal = self.__is_terminal(state)

		return state, reward, terminal

	def __is_terminal(self, state):
		return state in (self.gold, self.hole)

	def render(self, final_q_table=None):
		def map_range(x, from_lo, from_hi, to_lo, to_hi):
			if from_hi - from_lo == 0:
				return to_hi
			return (x - from_lo) / (from_hi - from_lo) * (to_hi - to_lo) + to_lo

		def shift_arrow(arrow, dx, dy):
			new_points = []
			for x, y in arrow:
				new_points.append((x + CELL_SIZE * (dx + 0.34), y + CELL_SIZE * (dy + 0.48)))
			return tuple(new_points)

		pg.init()
		pg.display.set_caption('Grid Environment')
		scene = pg.display.set_mode((self.size * CELL_SIZE, self.size * CELL_SIZE))
		font = pg.font.SysFont('consolas', 18)

		smallest_grid_q = biggest_grid_q = None
		if final_q_table:
			# Smallest and biggest best state-action values of grid
			smallest_grid_q = min(max(action_vals) for state, action_vals in final_q_table.items()
				if not self.__is_terminal(state))
			biggest_grid_q = max(max(action_vals) for state, action_vals in final_q_table.items()
				if not self.__is_terminal(state))

		for y in range(self.size):
			for x in range(self.size):
				best_action_val = None

				if not final_q_table or (y, x) not in final_q_table:
					col = (0, 0, 0)
				else:
					best_action_val = max(final_q_table[y, x])
					c = map_range(best_action_val, smallest_grid_q, biggest_grid_q, 20, 120)
					col = (c, c, c)

				match (y, x):
					case self.start: col = START_COL
					case self.gold: col = GOLD_COL
					case self.hole: col = HOLE_COL

				pg.draw.rect(scene, col, pg.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

				if self.__is_terminal((y, x)):
					terminal_lbl = font.render('Terminal', True, TEXT_COL)
					state_value_lbl = font.render(f'({self.rewards[(y, x)]})', True, TEXT_COL)
					lbl_rect1 = terminal_lbl.get_rect(center=((x + 0.5) * CELL_SIZE, (y + 0.33) * CELL_SIZE))
					lbl_rect2 = state_value_lbl.get_rect(center=((x + 0.5) * CELL_SIZE, (y + 0.66) * CELL_SIZE))
					scene.blit(terminal_lbl, lbl_rect1)
					scene.blit(state_value_lbl, lbl_rect2)
				elif best_action_val is not None:
					# Draw label of best action value of state
					act_value_lbl = font.render(f'{best_action_val:.3f}', True, TEXT_COL)
					lbl_rect = act_value_lbl.get_rect(center=((x + 0.5) * CELL_SIZE, (y + 0.33) * CELL_SIZE))
					scene.blit(act_value_lbl, lbl_rect)

					# Draw arrow pointing in optimal direction (best policy)
					best_action = final_q_table[y, x].argmax()
					if best_action == NORTH: pg.draw.polygon(scene, ARROW_COL, shift_arrow(NORTH_ARROW, x, y))
					if best_action == EAST: pg.draw.polygon(scene, ARROW_COL, shift_arrow(EAST_ARROW, x, y))
					if best_action == SOUTH: pg.draw.polygon(scene, ARROW_COL, shift_arrow(SOUTH_ARROW, x, y))
					if best_action == WEST: pg.draw.polygon(scene, ARROW_COL, shift_arrow(WEST_ARROW, x, y))
				elif (y, x) == self.start:
					start_lbl = font.render('Start', True, TEXT_COL)
					lbl_rect = start_lbl.get_rect(center=((x + 0.5) * CELL_SIZE, (y + 0.5) * CELL_SIZE))
					scene.blit(start_lbl, lbl_rect)

		pg.display.update()

	def generate(self):
		# y before x, as 2D arrays are row-major
		all_coords = np.array([[y, x] for y in range(self.size) for x in range(self.size)])
		random_coords = all_coords[np.random.choice(len(all_coords), size=3, replace=False)]

		self.start, self.gold, self.hole = [tuple(pair) for pair in random_coords]
		self.rewards = {self.gold: 10, self.hole: -10}

		self.render()
