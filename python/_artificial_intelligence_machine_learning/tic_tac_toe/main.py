"""
Tic Tac Toe player using minimax algorithm with alpha-beta pruning

Author: Sam Barba
Created 08/02/2022
"""

import sys

import pygame as pg


AI = 'x'
HUMAN = 'o'
TIE = 't'
BOARD_SIZE = 3
CELL_SIZE = 120
GRID_OFFSET = 80
FOREGROUND = (224, 224, 224)

board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
status_text = "Your turn (or 'A' to make AI go first)"


def handle_mouse_click(y, x):
	global status_text

	y = (y - GRID_OFFSET) // CELL_SIZE
	x = (x - GRID_OFFSET) // CELL_SIZE
	if y not in range(BOARD_SIZE) \
		or x not in range(BOARD_SIZE) \
		or board[y][x] \
		or find_winner():
		return

	board[y][x] = HUMAN

	result = find_winner()
	# No point checking if human wins...
	if result == TIE: status_text = "It's a tie! Click to reset"
	if not result: status_text = "AI's turn (x)"

	draw_grid()

	if not result:  # AI's turn
		pg.time.delay(1000)
		make_best_ai_move()
		draw_grid()


def find_winner():
	# Check rows and columns
	for i in range(BOARD_SIZE):
		row = board[i]
		if len(set(row)) == 1 and row[0]:
			return row[0]
		col = [row[i] for row in board]
		if len(set(col)) == 1 and col[0]:
			return col[0]

	# Check diagonals
	main_diagonal = [board[i][i] for i in range(BOARD_SIZE)]
	if len(set(main_diagonal)) == 1 and main_diagonal[0]:
		return main_diagonal[0]
	right_diagonal = [board[i][BOARD_SIZE - i - 1] for i in range(BOARD_SIZE)]
	if len(set(right_diagonal)) == 1 and right_diagonal[0]:
		return right_diagonal[0]

	free_spots = any(token is None for row in board for token in row)

	return None if free_spots else TIE


def make_best_ai_move():
	def minimax(is_maximising, depth, alpha, beta):
		result = find_winner()
		if result == AI: return 1
		if result == HUMAN: return -1
		if result == TIE: return 0

		best_score = -2 if is_maximising else 2

		for y in range(BOARD_SIZE):
			for x in range(BOARD_SIZE):
				if not board[y][x]:
					if is_maximising:
						board[y][x] = AI
						best_score = max(best_score, minimax(False, depth + 1, alpha, beta))
						alpha = max(alpha, best_score)
					else:
						board[y][x] = HUMAN
						best_score = min(best_score, minimax(True, depth + 1, alpha, beta))
						beta = min(beta, best_score)
					board[y][x] = None
					if beta <= alpha: return best_score / depth

		# Prefer shallower results over deeper results
		return best_score / depth


	global status_text

	best_score = -2
	best_y = best_x = 0

	for y in range(BOARD_SIZE):
		for x in range(BOARD_SIZE):
			if not board[y][x]:
				board[y][x] = AI
				score = minimax(False, 1, -2, 2)
				board[y][x] = None

				if score > best_score:
					best_score = score
					best_y, best_x = y, x

	board[best_y][best_x] = AI

	result = find_winner()
	if result == AI: status_text = 'AI wins! Click to reset'
	elif result == TIE: status_text = "It's a tie! Click to reset"
	elif not result: status_text = 'Your turn (o)'


def draw_grid():
	scene.fill((16, 16, 16))

	status_lbl = status_font.render(status_text, True, FOREGROUND)
	lbl_rect = status_lbl.get_rect(center=(BOARD_SIZE * CELL_SIZE / 2 + GRID_OFFSET, 40))
	scene.blit(status_lbl, lbl_rect)

	for y in range(BOARD_SIZE):
		for x in range(BOARD_SIZE):
			token = board[y][x]
			if not token: continue

			colour = (220, 20, 20) if token == AI else (20, 120, 220)
			cell_lbl = token_font.render(token, True, colour)
			lbl_rect = cell_lbl.get_rect(center=((x + 0.5) * CELL_SIZE + GRID_OFFSET,
				(y + 0.5) * CELL_SIZE + GRID_OFFSET))
			scene.blit(cell_lbl, lbl_rect)

	# Grid lines
	for y in range(GRID_OFFSET, BOARD_SIZE * CELL_SIZE + GRID_OFFSET + 1, CELL_SIZE):
		pg.draw.line(scene, FOREGROUND, (y, GRID_OFFSET), (y, BOARD_SIZE * CELL_SIZE + GRID_OFFSET))
		pg.draw.line(scene, FOREGROUND, (GRID_OFFSET, y), (BOARD_SIZE * CELL_SIZE + GRID_OFFSET, y))

	pg.display.update()


if __name__ == '__main__':
	pg.init()
	pg.display.set_caption('Tic Tac Toe')
	scene = pg.display.set_mode((BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET, BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET))
	status_font = pg.font.SysFont('consolas', 16)
	token_font = pg.font.SysFont('consolas', 140)

	draw_grid()

	while True:
		for event in pg.event.get():
			match event.type:
				case pg.QUIT: sys.exit()
				case pg.MOUSEBUTTONDOWN:
					if find_winner():  # Click to reset if game over
						board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
						status_text = "Your turn (or 'A' to make AI go first)"
						draw_grid()
					else:
						y, x = event.pos
						handle_mouse_click(x, y)
				case pg.KEYDOWN:
					if event.key == pg.K_a:  # Make AI play first
						# If no moves have been played yet
						if all(cell is None for row in board for cell in row):
							make_best_ai_move()
							draw_grid()
