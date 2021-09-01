"""
Snake Game

Author: Sam Barba
Created 10/02/2022

Controls:
WASD or arrow keys: move snake around
"""

import pygame as pg
import random
import sys
from time import sleep

ROWS, COLS = 30, 40
CELL_SIZE = 25
FPS = 8

# Directions
NORTH = [0, -1]
SOUTH = [0, 1]
EAST = [1, 0]
WEST = [-1, 0]

snake = food_pos = heading = score = game_over = None

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def setup():
	global snake, heading, score, game_over

	# x, y coords of head of snake are snake[0]
	snake = [[COLS // 2 - i, ROWS // 2 - 1] for i in range(3)]
	heading = EAST
	score = 0
	game_over = False
	generate_food()

def move_snake():
	snake_copy = snake[:]
	new_head = [s + h for s, h in zip(snake[0], heading)]

	for i in range(1, len(snake)):
		snake[i] = snake_copy[i - 1]

	snake[0] = new_head

def generate_food():
	global food_pos

	food_pos = [random.randrange(COLS), random.randrange(ROWS)]
	while food_pos in snake:
		food_pos = [random.randrange(COLS), random.randrange(ROWS)]

def check_eaten_food():
	global score

	if snake[0] == food_pos:
		score += 1
		snake.append(snake[-1])
		generate_food()

def check_game_over():
	global game_over

	head = snake[0]

	# Game over if snake has headed out of grid, or into its tail
	game_over = head[0] not in range(COLS) \
		or head[1] not in range(ROWS) \
		or head in snake[1:]

def draw_grid():
	scene.fill((0, 0, 0))

	# Draw snake (light blue head, dark blue tail or all red if game over)
	head_x, head_y = snake[0]
	head_colour = (255, 0, 0) if game_over else (0, 128, 255)
	tail_colour = (255, 0, 0) if game_over else (0, 0, 255)
	pg.draw.rect(scene, head_colour, pg.Rect(head_x * CELL_SIZE, head_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
	for x, y in snake[1:]:
		pg.draw.rect(scene, tail_colour, pg.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

	# Draw food
	pg.draw.rect(scene, (255, 128, 0), pg.Rect(food_pos[0] * CELL_SIZE, food_pos[1] * CELL_SIZE,
		CELL_SIZE, CELL_SIZE))

	# Score label
	score_lbl = font.render(f'Score: {score}', True, (220, 220, 220))
	scene.blit(score_lbl, (10, 10))

	pg.display.update()

	if game_over:
		sleep(1)
		scene.fill((0, 0, 0))
		lbl_texts = ['GAME OVER', f'Score: {score}', 'Press any key to reset']
		for idx, lbl_text in enumerate(lbl_texts):
			lbl = font.render(lbl_text, True, (0, 180, 0))
			offset = 40 * (idx - 1)
			lbl_rect = lbl.get_rect(center=(CELL_SIZE * COLS / 2, CELL_SIZE * ROWS / 2 + offset))
			scene.blit(lbl, lbl_rect)
		pg.display.update()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	pg.init()
	pg.display.set_caption('Snake Game')
	scene = pg.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE))
	clock = pg.time.Clock()
	font = pg.font.SysFont('consolas', 20)

	setup()

	while True:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				sys.exit(0)
			elif event.type == pg.KEYDOWN:
				if game_over:  # Press any key to reset
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

		if not game_over:
			draw_grid()
			move_snake()
			check_game_over()
			check_eaten_food()

			if game_over: draw_grid()  # Draw 'GAME OVER' screen

		clock.tick(FPS)
