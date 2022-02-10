# Snake Game
# Author: Sam Barba
# Created 10/02/2022

# WASD or arrow keys: Move snake around

import pygame as pg
import random
import sys
from time import sleep

ROWS = 30
COLS = 40
CELL_SIZE = 25
FPS = 8

# Directions
NORTH = [0, -1]
SOUTH = [0, 1]
EAST = [1, 0]
WEST = [-1, 0]

snake = None
foodPos = None
heading = None
score = None
gameOver = None

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def setup():
	global snake, heading, score, gameOver

	# x, y coords of head of snake are snake[0]
	snake = [[COLS // 2 - i, ROWS // 2 - 1] for i in range(3)]
	heading = EAST
	score = 0
	gameOver = False
	generateFood()

def moveSnake():
	snakeCopy = snake[:]
	newHead = [s + h for s, h in zip(snake[0], heading)]

	for i in range(1, len(snake)):
		snake[i] = snakeCopy[i - 1]

	snake[0] = newHead

def generateFood():
	global foodPos

	foodPos = [random.randrange(COLS), random.randrange(ROWS)]
	while foodPos in snake:
		foodPos = [random.randrange(COLS), random.randrange(ROWS)]

def checkEatenFood():
	global score

	if snake[0] == foodPos:
		score += 1
		snake.append(snake[-1])
		generateFood()

def checkGameOver():
	global gameOver

	head = snake[0]

	# Game over if snake has headed out of grid, or into its tail
	gameOver = head[0] not in range(COLS) \
		or head[1] not in range(ROWS) \
		or head in snake[1:]

def drawGrid():
	scene.fill((0, 0, 0))
	font = pg.font.SysFont("consolas", 20)

	# Draw snake (light blue head, dark blue tail or all red if game over)
	headX, headY = snake[0]
	headColour = (255, 0, 0) if gameOver else (0, 128, 255)
	tailColour = (255, 0, 0) if gameOver else (0, 0, 255)
	pg.draw.rect(scene, headColour, pg.Rect(headX * CELL_SIZE, headY * CELL_SIZE, CELL_SIZE, CELL_SIZE))
	for x, y in snake[1:]:
		pg.draw.rect(scene, tailColour, pg.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

	# Draw food
	pg.draw.rect(scene, (255, 128, 0), pg.Rect(foodPos[0] * CELL_SIZE, foodPos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

	# Score label
	scoreLbl = font.render(f"Score: {score}", True, (220, 220, 220))
	scene.blit(scoreLbl, (10, 10))

	pg.display.flip()

	if gameOver:
		sleep(1)
		scene.fill((0, 0, 0))
		lblTexts = ["GAME OVER", f"Score: {score}", "Press any key to reset"]
		for idx, lblText in enumerate(lblTexts):
			lbl = font.render(lblText, True, (0, 180, 0))
			offset = 40 * (idx - 1)
			lblRect = lbl.get_rect(center=(CELL_SIZE * COLS / 2, CELL_SIZE * ROWS / 2 + offset))
			scene.blit(lbl, lblRect)
		pg.display.flip()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

pg.init()
pg.display.set_caption("Snake Game")
scene = pg.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE))
clock = pg.time.Clock()

setup()

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)
		elif event.type == pg.KEYDOWN:
			if gameOver: # Press any key to reset
				setup()
			elif event.key in (pg.K_w, pg.K_UP) and heading != SOUTH:
				heading = NORTH
			elif event.key in (pg.K_s, pg.K_DOWN) and heading != NORTH:
				heading = SOUTH
			elif event.key in (pg.K_d, pg.K_RIGHT) and heading != WEST:
				heading = EAST
			elif event.key in (pg.K_a, pg.K_LEFT) and heading != EAST:
				heading = WEST
			break

	if not gameOver:
		drawGrid()
		moveSnake()
		checkGameOver()
		checkEatenFood()

		if gameOver: drawGrid() # Draw "GAME OVER" screen

	clock.tick(FPS)
