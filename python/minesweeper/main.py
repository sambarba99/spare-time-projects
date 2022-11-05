"""
Minesweeper

Author: Sam Barba
Created 27/09/2021
"""

import sys

import pygame as pg
import random

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
	def __init__(self, i, j):
		self.i = i
		self.j = j
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
			for i_offset in range(-1, 2):
				for j_offset in range(-1, 2):
					check_i = self.i + i_offset
					check_j = self.j + j_offset
					if check_i in range(ROWS) and check_j in range(COLS) \
						and minefield[check_i][check_j].is_mine:
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
		for i_offset in range(-1, 2):
			for j_offset in range(-1, 2):
				i = self.i + i_offset
				j = self.j + j_offset
				if i in range(ROWS) and j in range(COLS) \
					and not minefield[i][j].is_flagged:
					minefield[i][j].reveal(False)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def setup_game():
	global minefield, flags_used_total, flags_used_correctly, game_over, status_text

	minefield = [[Cell(i, j) for j in range(COLS)] for i in range(ROWS)]

	flags_used_total = flags_used_correctly = 0
	game_over = False

	all_coords = [(i, j) for i in range(ROWS) for j in range(COLS)]
	mine_coords = random.sample(all_coords, N_MINES)

	for i, j in mine_coords:
		minefield[i][j].is_mine = True

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

	for i in range(ROWS):
		for j in range(COLS):
			minefield[i][j].reveal(won)

	game_over = True
	status_text = 'YOU WIN! Click to reset.' if won else 'GAME OVER. Click to reset.'

def draw_grid():
	scene.fill(BACKGROUND)
	font = pg.font.SysFont('consolas', 16)

	for i in range(ROWS):
		for j in range(COLS):
			pg.draw.rect(scene, minefield[i][j].colour, pg.Rect(j * CELL_SIZE + GRID_OFFSET,
				i * CELL_SIZE + GRID_OFFSET, CELL_SIZE, CELL_SIZE))
			cell_lbl = font.render(minefield[i][j].text, True, LABEL_FOREGROUND)
			lbl_rect = cell_lbl.get_rect(center=((j + 0.5) * CELL_SIZE + GRID_OFFSET,
			(i + 0.5) * CELL_SIZE + GRID_OFFSET))
			scene.blit(cell_lbl, lbl_rect)

	status_lbl = font.render(status_text, True, LABEL_FOREGROUND)
	lbl_rect = status_lbl.get_rect(center=(COLS * CELL_SIZE / 2 + GRID_OFFSET, 35))
	scene.blit(status_lbl, lbl_rect)

	# Grid lines
	for j in range(GRID_OFFSET, COLS * CELL_SIZE + GRID_OFFSET + 1, CELL_SIZE):
		pg.draw.line(scene, BACKGROUND, (j, GRID_OFFSET), (j, ROWS * CELL_SIZE + GRID_OFFSET))
	for i in range(GRID_OFFSET, ROWS * CELL_SIZE + GRID_OFFSET + 1, CELL_SIZE):
		pg.draw.line(scene, BACKGROUND, (GRID_OFFSET, i), (COLS * CELL_SIZE + GRID_OFFSET, i))

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
			match event.type:
				case pg.QUIT: sys.exit()
				case pg.MOUSEBUTTONDOWN:
					if game_over:  # Reset
						setup_game()
					else:
						j, i = event.pos
						i = (i - GRID_OFFSET) // CELL_SIZE
						j = (j - GRID_OFFSET) // CELL_SIZE
						if i in range(ROWS) and j in range(COLS):
							minefield[i][j].handle_mouse_click(event.button)

					draw_grid()
