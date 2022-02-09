# Minesweeper
# Author: Sam Barba
# Created 27/09/2021

import pygame as pg
import random
import sys

CELL_SIZE = 22
ROWS = 35
COLS = 60
GRID_OFFSET = 60
NUM_MINES = round(ROWS * COLS * 0.1)
BACKGROUND = (20, 20, 20)
LABEL_FOREGROUND = (220, 220, 220)
CELL_UNCLICKED = (80, 80, 80)
CELL_FLAGGED = (255, 160, 0)
MINE_WON = (0, 144, 0)
MINE_LOST = (200, 20, 20)

minefield = None
flagsUsedTotal = 0
flagsUsedCorrectly = 0
gameOver = False
statusText = ""

# ---------------------------------------------------------------------------------------------------- #
# ---------------------------------------------  CLASSES  -------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

class Cell:
	def __init__(self, y, x): # Y before X, as 2D arrays are row-major
		self.y = y
		self.x = x
		self.isMine = False
		self.isRevealed = False
		self.isFlagged = False
		self.colour = CELL_UNCLICKED
		self.text = ""

	def handleMouseClick(self, eventButton):
		if gameOver or self.isRevealed: return

		if eventButton == 1 and not self.isFlagged: # Left-click
			if self.isMine:
				endGame(False)
			else:
				self.reveal(False)
				checkWin()
		elif eventButton == 3: # Right-click
			self.toggleFlag()
			checkWin()

	def reveal(self, won):
		if self.isRevealed: return

		self.isRevealed = True

		if self.isMine:
			self.colour = MINE_WON if won else MINE_LOST
		else:
			self.colour = BACKGROUND

			n = self.countNeighbourMines()
			if n:
				self.text = str(n)
			else:
				# Recursively reveal cells with 0 neighbouring mines
				self.floodReveal()

	def countNeighbourMines(self):
		n = 0
		for yOffset in range(-1, 2):
			for xOffset in range(-1, 2):
				checkY = self.y + yOffset
				checkX = self.x + xOffset
				if checkY in range(ROWS) and checkX in range(COLS) \
					and minefield[checkY][checkX].isMine:
					n += 1
		return n

	def floodReveal(self):
		for yOffset in range(-1, 2):
			for xOffset in range(-1, 2):
				y = self.y + yOffset
				x = self.x + xOffset
				if y in range(ROWS) and x in range(COLS) \
					and not minefield[y][x].isFlagged:
					minefield[y][x].reveal(False)

	def toggleFlag(self):
		global flagsUsedTotal, flagsUsedCorrectly

		if self.isFlagged:
			flagsUsedTotal -= 1
			if self.isMine:
				flagsUsedCorrectly -= 1
			self.colour = CELL_UNCLICKED
			self.isFlagged = False
		elif NUM_MINES - flagsUsedTotal > 0: # If there are flags left to use
			flagsUsedTotal += 1
			if self.isMine:
				flagsUsedCorrectly += 1
			self.colour = CELL_FLAGGED
			self.isFlagged = True

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def setupGame():
	global minefield, flagsUsedTotal, flagsUsedCorrectly, gameOver, statusText

	minefield = [[Cell(y, x) for x in range(COLS)] for y in range(ROWS)]

	flagsUsedTotal = flagsUsedCorrectly = 0
	gameOver = False

	allCoords = [(y, x) for x in range(COLS) for y in range(ROWS)]
	mineCoords = random.sample(allCoords, NUM_MINES)

	for y, x in mineCoords:
		minefield[y][x].isMine = True

	statusText = f"Flags left: {NUM_MINES - flagsUsedTotal}"

# Win if: All mines are correctly flagged; and all non-mine cells are revealed
def checkWin():
	global statusText

	allNonMinesRevealed = all(cell.isRevealed for row in minefield for cell in row if not cell.isMine)

	if flagsUsedCorrectly == NUM_MINES and allNonMinesRevealed:
		endGame(True)
	else:
		statusText = f"Flags left: {NUM_MINES - flagsUsedTotal}"

def endGame(won):
	global gameOver, statusText

	for y in range(ROWS):
		for x in range(COLS):
			minefield[y][x].reveal(won)

	gameOver = True
	statusText = "YOU WIN! Click to reset." if won else "GAME OVER. Click to reset."

def drawGrid():
	scene.fill(BACKGROUND)
	font = pg.font.SysFont("consolas", 16)

	for y in range(ROWS):
		for x in range(COLS):
			pg.draw.rect(scene, minefield[y][x].colour, pg.Rect(x * CELL_SIZE + GRID_OFFSET, y * CELL_SIZE + GRID_OFFSET, CELL_SIZE, CELL_SIZE))
			cellLbl = font.render(minefield[y][x].text, True, LABEL_FOREGROUND)
			scene.blit(cellLbl, (x * CELL_SIZE + GRID_OFFSET + 7, y * CELL_SIZE + GRID_OFFSET + 5))

	statusLabel = font.render(statusText, True, LABEL_FOREGROUND)
	scene.blit(statusLabel, (GRID_OFFSET, GRID_OFFSET // 2))

	# Grid lines
	for x in range(GRID_OFFSET, COLS * CELL_SIZE + GRID_OFFSET + 1, CELL_SIZE):
		pg.draw.line(scene, BACKGROUND, (x, GRID_OFFSET), (x, ROWS * CELL_SIZE + GRID_OFFSET))
	for y in range(GRID_OFFSET, ROWS * CELL_SIZE + GRID_OFFSET + 1, CELL_SIZE):
		pg.draw.line(scene, BACKGROUND, (GRID_OFFSET, y), (COLS * CELL_SIZE + GRID_OFFSET, y))

	pg.display.flip()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

pg.init()
pg.display.set_caption("Minesweeper")
scene = pg.display.set_mode((COLS * CELL_SIZE + 2 * GRID_OFFSET, ROWS * CELL_SIZE + 2 * GRID_OFFSET))

sys.setrecursionlimit(ROWS * COLS)
setupGame()
drawGrid()

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)
		elif event.type == pg.MOUSEBUTTONDOWN:
			if gameOver: # Reset
				setupGame()
			else:
				x, y = event.pos
				y = (y - GRID_OFFSET) // CELL_SIZE
				x = (x - GRID_OFFSET) // CELL_SIZE
				if y in range(ROWS) and x in range(COLS):
					minefield[y][x].handleMouseClick(event.button)

			drawGrid()
