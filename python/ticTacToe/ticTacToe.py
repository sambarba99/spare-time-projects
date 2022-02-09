# Tic Tac Toe player using minimax algorithm with alpha-beta pruning
# Author: Sam Barba
# Created 08/02/2022

# A: Make AI play first move
# R: Reset

import pygame as pg
import sys
from time import sleep

AI = "x"
HUMAN = "o"
TIE = "t"
BOARD_SIZE = 3
CELL_SIZE = 120
GRID_OFFSET = 80
FOREGROUND = (220, 220, 220)

board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
statusText = "Your turn! (Or 'A' to make AI go first)"

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def handleMouseClick(y, x):
	y = (y - GRID_OFFSET) // CELL_SIZE
	x = (x - GRID_OFFSET) // CELL_SIZE
	if y not in range(BOARD_SIZE) or x not in range(BOARD_SIZE) \
		or board[y][x] is not None \
		or findWinner() is not None:
		return

	global statusText

	board[y][x] = HUMAN

	result = findWinner()
	# No point checking if human wins...
	if result == TIE: statusText = "It's a tie! 'R' to reset"
	if result is None: statusText = "AI's turn (x)"

	drawGrid()

	if result is None: # AI's turn
		sleep(1)
		makeBestAiMove()
		drawGrid()

def findWinner():
	# Check rows and columns
	for i in range(BOARD_SIZE):
		row = board[i]
		if len(set(row)) == 1 and row[0] is not None:
			return row[0]
		col = [row[i] for row in board]
		if len(set(col)) == 1 and col[0] is not None:
			return col[0]

	# Check diagonals
	mainDiagonal = [board[i][i] for i in range(BOARD_SIZE)]
	if len(set(mainDiagonal)) == 1 and mainDiagonal[0] is not None:
		return mainDiagonal[0]
	rightDiagonal = [board[i][BOARD_SIZE - i - 1] for i in range(BOARD_SIZE)]
	if len(set(rightDiagonal)) == 1 and rightDiagonal[0] is not None:
		return rightDiagonal[0]

	freeSpots = sum(token is None for row in board for token in row)

	return TIE if freeSpots == 0 else None

def makeBestAiMove():
	bestScore = -2
	bestY = bestX = 0

	for y in range(BOARD_SIZE):
		for x in range(BOARD_SIZE):
			if board[y][x] is None:
				board[y][x] = AI
				score = minimax(1, -2, 2, False)
				board[y][x] = None

				if score > bestScore:
					bestScore = score
					bestY, bestX = y, x

	board[bestY][bestX] = AI

	result = findWinner()
	global statusText
	if result == AI: statusText = "AI wins! 'R' to reset"
	if result == TIE: statusText = "It's a tie! 'R' to reset"
	if result is None: statusText = "Your turn (o)! 'R' to reset"

def minimax(depth, alpha, beta, maximising):
	result = findWinner()
	if result == AI: return 1
	if result == HUMAN: return -1
	if result == TIE: return 0

	bestScore = -2 if maximising else 2

	for y in range(BOARD_SIZE):
		for x in range(BOARD_SIZE):
			if board[y][x] is None:
				if maximising:
					board[y][x] = AI
					bestScore = max(minimax(depth + 1, alpha, beta, False), bestScore)
					alpha = max(alpha, bestScore)
					if beta <= alpha:
						board[y][x] = None
						break
				else:
					board[y][x] = HUMAN
					bestScore = min(minimax(depth + 1, alpha, beta, True), bestScore)
					beta = min(beta, bestScore)
					if beta <= alpha:
						board[y][x] = None
						break
				board[y][x] = None

	# Prefer shallower results over deeper results
	return bestScore / depth

def drawGrid():
	scene.fill((20, 20, 20))
	statusFont = pg.font.SysFont("consolas", 16)
	tokenFont = pg.font.SysFont("consolas", 140)

	statusLbl = statusFont.render(statusText, True, FOREGROUND)
	scene.blit(statusLbl, (GRID_OFFSET, 35))

	for y in range(BOARD_SIZE):
		for x in range(BOARD_SIZE):
			token = board[y][x]
			if token is None: continue

			colour = (220, 20, 20) if token == AI else (20, 120, 220)
			cellLbl = tokenFont.render(token, True, colour)
			scene.blit(cellLbl, (x * CELL_SIZE + GRID_OFFSET + 22, y * CELL_SIZE + GRID_OFFSET - 9))

	# Grid lines
	for i in range(GRID_OFFSET, BOARD_SIZE * CELL_SIZE + GRID_OFFSET + 1, CELL_SIZE):
		pg.draw.line(scene, FOREGROUND, (i, GRID_OFFSET), (i, BOARD_SIZE * CELL_SIZE + GRID_OFFSET))
		pg.draw.line(scene, FOREGROUND, (GRID_OFFSET, i), (BOARD_SIZE * CELL_SIZE + GRID_OFFSET, i))

	pg.display.flip()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

pg.init()
pg.display.set_caption("Tic Tac Toe")
scene = pg.display.set_mode((BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET, BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET))

drawGrid()

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)

		elif event.type == pg.MOUSEBUTTONDOWN:
			x, y = event.pos
			handleMouseClick(y, x)

		elif event.type == pg.KEYDOWN:
			if event.key == pg.K_a: # Make AI play first
				# If no moves have been played yet
				if all(cell is None for row in board for cell in row):
					makeBestAiMove()
					drawGrid()
			elif event.key == pg.K_r: # Reset
				board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
				statusText = "Your turn! (Or 'A' to make AI go first)"
				drawGrid()
