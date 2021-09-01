"""
Minesweeper

Author: Sam Barba
Created 27/09/2021
"""

import pygame as pg
import random
import sys

ROWS, COLS = 35, 60
CELL_SIZE = 22
GRID_OFFSET = 60
N_MINES = int(ROWS * COLS * 0.1)
BACKGROUND = (20, 20, 20)
LABEL_FOREGROUND = (220, 220, 220)
CELL_UNCLICKED = (80, 80, 80)
CELL_FLAGGED = (255, 160, 0)
MINE_WON = (0, 144, 0)
MINE_LOST = (255, 20, 20)

minefield = None
flags_used_total = flags_used_correctly = 0
game_over = False
status_text = ''

# ---------------------------------------------------------------------------------------------------- #
# ---------------------------------------------  CLASSES  -------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

class Cell:
	def __init__(self, y, x):  # Y before X, as 2D arrays are row-major
		self.y = y
		self.x = x
		self.is_mine = False
		self.is_revealed = False
		self.is_flagged = False
		self.colour = CELL_UNCLICKED
		self.text = ''

	def handle_mouse_click(self, event_button):
		def toggle_flag():
			global flags_used_total, flags_used_correctly

			if self.is_flagged:
				flags_used_total -= 1
				if self.is_mine:
					flags_used_correctly -= 1
				self.colour = CELL_UNCLICKED
				self.is_flagged = False
			elif N_MINES - flags_used_total > 0:  # If there are flags left to use
				flags_used_total += 1
				if self.is_mine:
					flags_used_correctly += 1
				self.colour = CELL_FLAGGED
				self.is_flagged = True

		if game_over or self.is_revealed: return

		if event_button == 1 and not self.is_flagged:  # Left-click
			if self.is_mine:
				end_game(False)
			else:
				self.reveal(False)
				check_win()
		elif event_button == 3:  # Right-click
			toggle_flag()
			check_win()

	def reveal(self, won):
		def count_neighbour_mines():
			n = 0
			for y_offset in range(-1, 2):
				for x_offset in range(-1, 2):
					check_y = self.y + y_offset
					check_x = self.x + x_offset
					if check_y in range(ROWS) and check_x in range(COLS) \
						and minefield[check_y][check_x].is_mine:
						n += 1
			return n

		if self.is_revealed: return

		self.is_revealed = True

		if self.is_mine:
			self.colour = MINE_WON if won else MINE_LOST
		else:
			self.colour = BACKGROUND

			n = count_neighbour_mines()
			if n:
				self.text = str(n)
			else:
				# Recursively reveal cells with 0 neighbouring mines
				self.flood_reveal()

	def flood_reveal(self):
		for y_offset in range(-1, 2):
			for x_offset in range(-1, 2):
				y = self.y + y_offset
				x = self.x + x_offset
				if y in range(ROWS) and x in range(COLS) \
					and not minefield[y][x].is_flagged:
					minefield[y][x].reveal(False)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def setup_game():
	global minefield, flags_used_total, flags_used_correctly, game_over, status_text

	minefield = [[Cell(y, x) for x in range(COLS)] for y in range(ROWS)]

	flags_used_total = flags_used_correctly = 0
	game_over = False

	all_coords = [(y, x) for y in range(ROWS) for x in range(COLS)]
	mine_coords = random.sample(all_coords, N_MINES)

	for y, x in mine_coords:
		minefield[y][x].is_mine = True

	status_text = f'Flags left: {N_MINES - flags_used_total}'

def check_win():
	"""Win if: all mines are correctly flagged; and all non-mine cells are revealed"""

	global status_text

	all_non_mines_revealed = all(cell.is_revealed for row in minefield for cell in row if not cell.is_mine)

	if flags_used_correctly == N_MINES and all_non_mines_revealed:
		end_game(True)
	else:
		status_text = f'Flags left: {N_MINES - flags_used_total}'

def end_game(won):
	global game_over, status_text

	for y in range(ROWS):
		for x in range(COLS):
			minefield[y][x].reveal(won)

	game_over = True
	status_text = 'YOU WIN! Click to reset.' if won else 'GAME OVER. Click to reset.'

def draw_grid():
	scene.fill(BACKGROUND)
	font = pg.font.SysFont('consolas', 16)

	for y in range(ROWS):
		for x in range(COLS):
			pg.draw.rect(scene, minefield[y][x].colour, pg.Rect(x * CELL_SIZE + GRID_OFFSET,
				y * CELL_SIZE + GRID_OFFSET, CELL_SIZE, CELL_SIZE))
			cell_lbl = font.render(minefield[y][x].text, True, LABEL_FOREGROUND)
			lbl_rect = cell_lbl.get_rect(center=((x + 0.5) * CELL_SIZE + GRID_OFFSET,
			(y + 0.5) * CELL_SIZE + GRID_OFFSET))
			scene.blit(cell_lbl, lbl_rect)

	status_lbl = font.render(status_text, True, LABEL_FOREGROUND)
	lbl_rect = status_lbl.get_rect(center=(COLS * CELL_SIZE / 2 + GRID_OFFSET, 35))
	scene.blit(status_lbl, lbl_rect)

	# Grid lines
	for x in range(GRID_OFFSET, COLS * CELL_SIZE + GRID_OFFSET + 1, CELL_SIZE):
		pg.draw.line(scene, BACKGROUND, (x, GRID_OFFSET), (x, ROWS * CELL_SIZE + GRID_OFFSET))
	for y in range(GRID_OFFSET, ROWS * CELL_SIZE + GRID_OFFSET + 1, CELL_SIZE):
		pg.draw.line(scene, BACKGROUND, (GRID_OFFSET, y), (COLS * CELL_SIZE + GRID_OFFSET, y))

	pg.display.update()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	pg.init()
	pg.display.set_caption('Minesweeper')
	scene = pg.display.set_mode((COLS * CELL_SIZE + 2 * GRID_OFFSET, ROWS * CELL_SIZE + 2 * GRID_OFFSET))

	sys.setrecursionlimit(ROWS * COLS)
	setup_game()
	draw_grid()

	while True:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				sys.exit(0)
			elif event.type == pg.MOUSEBUTTONDOWN:
				if game_over:  # Reset
					setup_game()
				else:
					x, y = event.pos
					y = (y - GRID_OFFSET) // CELL_SIZE
					x = (x - GRID_OFFSET) // CELL_SIZE
					if y in range(ROWS) and x in range(COLS):
						minefield[y][x].handle_mouse_click(event.button)

				draw_grid()
