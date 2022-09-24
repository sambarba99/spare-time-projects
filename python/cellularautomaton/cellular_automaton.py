"""
Elementary Cellular Automaton demo

Author: Sam Barba
Created 23/09/2021
"""

import pygame as pg
import sys
from time import sleep

IMG_SIZE = 369
CELL_SIZE = 2

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def set_ruleset(n):
	if n not in range(1, 256):
		raise ValueError('Rule no. must be 1 - 255')

	ruleset = [0] * 8
	i = 7
	while n:
		ruleset[i] = n % 2
		n //= 2
		i -= 1

	return ruleset

def generate_plot(ruleset):
	gen = [0] * IMG_SIZE
	gen[IMG_SIZE // 2] = 1  # Turn on centre pixel of first generation
	plot = [None] * IMG_SIZE

	for y in range(IMG_SIZE):  # Y before X, as 2D arrays are row-major
		plot[y] = gen[:]
		next_gen = [0] * IMG_SIZE

		for x in range(IMG_SIZE):
			left = 0 if x == 0 else gen[x - 1]
			centre = gen[x]
			right = 0 if x == IMG_SIZE - 1 else gen[x + 1]
			next_gen[x] = ruleset[7 - (4 * left + 2 * centre + right)]

		gen = next_gen[:]

	scene.fill((20, 20, 20))
	pg.display.update()
	sleep(1)
	for y in range(IMG_SIZE):
		for x in range(IMG_SIZE):
			c = (220, 220, 220) if plot[y][x] == 1 else (20, 20, 20)
			pg.draw.rect(scene, c, pg.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
		pg.display.update()
		sleep(0.01)

		for event in pg.event.get():
			if event.type == pg.QUIT:
				sys.exit()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	pg.init()
	scene = pg.display.set_mode((IMG_SIZE * CELL_SIZE, IMG_SIZE * CELL_SIZE))

	# Interesting rules
	rules = [18, 30, 45, 54, 57, 73, 105, 151, 153, 161]
	i = 0

	while True:
		ruleset = set_ruleset(rules[i])
		pg.display.set_caption(f'Elementary Cellular Automaton (rule {rules[i]}: {ruleset})')
		generate_plot(ruleset)
		i = (i + 1) % len(rules)
		sleep(2)
