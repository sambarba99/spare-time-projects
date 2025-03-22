"""
Game class

Author: Sam Barba
Created 28/01/2025
"""

from dataclasses import dataclass, field
import random
import sys

import pygame as pg

from solver import Solver


NUM_IMGS = [pg.image.load(f'./imgs/{i}.png') for i in range(9)]
FLAG_IMG = pg.image.load('./imgs/flag.png')
FLAG_INCORRECT_IMG = pg.image.load('./imgs/flag_incorrect.png')
MINE_IMG = pg.image.load('./imgs/mine.png')
MINE_RED_IMG = pg.image.load('./imgs/mine_red.png')
UNOPENED_IMG = pg.image.load('./imgs/unopened.png')
IMG_PX_SIZE = 30
PAD_TOP = 75
FPS = 12


@dataclass
class Cell:
	y: int
	x: int
	is_mine: bool = False
	is_flagged: bool = False
	is_open: bool = False
	neighbours: list = field(default_factory=list)
	num_surrounding_mines: int = None

	# For solver.py
	is_edge: bool = False      # If this cell is an edge (i.e. adjacent to at least 1 open cell)
	num_edges: int = 0         # No. edges adjacent to this cell
	num_mine_configs: int = 0  # No. possible configurations in which this cell is a mine
	mine_prob: int = None      # % probability of being a mine (0-100)


class Game:
	def __init__(self, rows, cols, num_mines, seed=None):
		assert 0 < num_mines < rows * cols

		self.rows = rows
		self.cols = cols
		self.scene_width = self.cols * IMG_PX_SIZE
		self.scene_height = self.rows * IMG_PX_SIZE + PAD_TOP
		self.num_mines = num_mines
		self.all_coords = [(y, x) for y in range(self.rows) for x in range(self.cols)]
		self.seed = seed

		self.grid = None
		self.done_first_click = None
		self.game_over = None
		self.flags_used = None
		self.num_closed_cells = None
		self.status = None
		self.clicked_mine_coords = None
		self.solver = None
		self.show_solver_probs = None
		self.setup()

		pg.init()
		pg.display.set_caption('Minesweeper')
		self.scene = pg.display.set_mode((self.scene_width, self.scene_height))
		self.clock = pg.time.Clock()
		self.font18 = pg.font.SysFont('consolas', 18)
		self.font14 = pg.font.SysFont('consolas', 14)

	def __iter__(self):
		for row in self.grid:
			for cell in row:
				yield cell

	def setup(self):
		if self.seed is not None:
			random.seed(self.seed)
		self.grid = [[Cell(y, x) for x in range(self.cols)] for y in range(self.rows)]
		self.done_first_click = False
		self.game_over = False
		self.flags_used = 0
		self.num_closed_cells = len(self.all_coords)
		self.status = str(self.num_mines)
		self.clicked_mine_coords = None
		self.solver = Solver(self)
		self.show_solver_probs = False

	def handle_click(self, y, x, mouse_button, is_screen_coords=False, render=True):
		assert mouse_button in ('left', 'right')

		if is_screen_coords:
			# Convert screen coords to grid coords
			y = (y - PAD_TOP) // IMG_PX_SIZE
			x //= IMG_PX_SIZE

		if (y, x) not in self.all_coords:
			# Ignore mouse clicks that are outside the minefield
			return

		if mouse_button == 'left' and not self.done_first_click:
			possible_mine_coords = self.all_coords[:]
			possible_mine_coords.remove((y, x))  # First clicked cell can't be a mine
			mine_coords = random.sample(possible_mine_coords, self.num_mines)
			for my, mx in mine_coords:
				self.grid[my][mx].is_mine = True
			self.done_first_click = True

			# Precompute some things
			for cell in self:
				cell.neighbours = self.get_neighbours(cell)
				cell.num_surrounding_mines = sum(neighbour.is_mine for neighbour in cell.neighbours)

		clicked_cell = self.grid[y][x]
		if clicked_cell.is_open:
			return

		if mouse_button == 'left' and not clicked_cell.is_flagged:
			if clicked_cell.is_mine:
				self.clicked_mine_coords = (y, x)
			else:
				self.open(clicked_cell)
		elif mouse_button == 'right':
			# Toggle flag
			if clicked_cell.is_flagged:
				clicked_cell.is_flagged = False
				self.flags_used -= 1
			elif self.flags_used < self.num_mines:
				# If there are flags left to use
				clicked_cell.is_flagged = True
				self.flags_used += 1
			self.status = str(self.num_mines - self.flags_used)

		if self.check_game_over():
			won = self.clicked_mine_coords is None
			self.status = f'{"YOU WIN!" if won else "GAME OVER."} Click to reset.'
			self.game_over = True

		if self.game_over:
			self.show_solver_probs = False

		if mouse_button == 'left' and self.show_solver_probs and render:
			self.solver.calculate_mine_probs()

		if render:
			self.render()

	def get_neighbours(self, cell):
		neighbours = []

		for ny in range(cell.y - 1, cell.y + 2):
			for nx in range(cell.x - 1, cell.x + 2):
				if (ny, nx) not in self.all_coords:
					continue
				if (ny, nx) == (cell.y, cell.x):
					continue
				neighbours.append(self.grid[ny][nx])

		return neighbours

	def open(self, cell):
		stack = [cell]

		while stack:
			cell_ = stack.pop()

			if cell_.is_open:
				continue

			cell_.is_open = True
			self.num_closed_cells -= 1

			if cell_.num_surrounding_mines == 0:
				# If no surrounding mines, add all non-open non-flagged neighbours to the stack
				for neighbour in cell_.neighbours:
					if not (neighbour.is_open or neighbour.is_flagged):
						stack.append(neighbour)

	def check_game_over(self):
		if self.clicked_mine_coords is not None:
			return True

		# Player wins if the only closed cells left are all mines
		# (if a mine has been opened, the player has clicked it by definition and thus lost)

		if self.num_closed_cells == self.num_mines:
			return True

		return False

	def solver_step(self):
		self.show_solver_probs = True

		# Unflag any incorrectly flagged cells
		for cell in self:
			if cell.is_flagged and cell.mine_prob != 100:
				self.handle_click(cell.y, cell.x, mouse_button='right', render=False)

		if not self.done_first_click:
			self.handle_click(1, 1, mouse_button='left')  # Start near corner
			return

		self.solver.calculate_mine_probs()

		# Find the cell with minimum probability of being a mine
		cell = min(
			(cell for cell in self if cell.mine_prob is not None),
			key=lambda cell: (cell.mine_prob, cell.y, cell.x)
		)
		self.handle_click(cell.y, cell.x, mouse_button='left')

		# Flag cells guaranteed to be mines
		for cell in self:
			if cell.mine_prob == 100 and not cell.is_flagged:
				self.handle_click(cell.y, cell.x, mouse_button='right')

	def auto_play(self):
		while not self.game_over:
			self.solver_step()

			for event in pg.event.get():
				if event.type == pg.QUIT:
					sys.exit()

	def render(self):
		self.scene.fill('black')

		if self.status.startswith('YOU WIN'):
			for cell in self:
				if cell.is_mine:
					cell.is_flagged = True
			status_colour = 'green'
		elif self.status.startswith('GAME OVER'):
			for cell in self:
				if cell.is_mine:
					cell.is_open = True
			status_colour = 'red'
			y, x = self.clicked_mine_coords
			self.scene.blit(MINE_RED_IMG, (x * IMG_PX_SIZE, y * IMG_PX_SIZE + PAD_TOP))
		else:
			status_colour = 'white'

		status_lbl = self.font18.render(self.status, True, status_colour)

		if self.game_over:
			lbl_rect = status_lbl.get_rect(center=(self.scene_width // 2, PAD_TOP // 2 + 2))
			self.scene.blit(status_lbl, lbl_rect)
		else:
			self.scene.blit(FLAG_IMG, (self.scene_width // 2 - 30, PAD_TOP // 2 - IMG_PX_SIZE // 2))
			self.scene.blit(status_lbl, (self.scene_width // 2 + 10, PAD_TOP // 2 - 8))

		for cell in self:
			if (cell.y, cell.x) == self.clicked_mine_coords:
				continue  # Already drew the red mine on screen
			if cell.is_open:
				img = MINE_IMG if cell.is_mine else NUM_IMGS[cell.num_surrounding_mines]
			else:
				if cell.is_flagged:
					img = FLAG_INCORRECT_IMG if self.game_over and not cell.is_mine else FLAG_IMG
				else:
					img = UNOPENED_IMG

			self.scene.blit(img, (cell.x * IMG_PX_SIZE, cell.y * IMG_PX_SIZE + PAD_TOP))

			if self.show_solver_probs and cell.mine_prob is not None:
				prob_lbl = self.font14.render(str(cell.mine_prob), True, 'black')
				lbl_rect = prob_lbl.get_rect(
					center=((cell.x + 0.5) * IMG_PX_SIZE, (cell.y + 0.5) * IMG_PX_SIZE + PAD_TOP)
				)
				self.scene.blit(prob_lbl, lbl_rect)

		pg.display.update()
		self.clock.tick(FPS)
