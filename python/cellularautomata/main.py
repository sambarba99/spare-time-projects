"""
Elementary cellular automata demo

Author: Sam Barba
Created 23/09/2021
"""

import sys
from time import sleep

import pygame as pg


IMG_SIZE = 369
CELL_SIZE = 2


def get_ruleset(n):
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

	for i in range(IMG_SIZE):
		plot[i] = gen[:]
		next_gen = [0] * IMG_SIZE

		for j in range(IMG_SIZE):
			left = 0 if j == 0 else gen[j - 1]
			centre = gen[j]
			right = 0 if j == IMG_SIZE - 1 else gen[j + 1]
			next_gen[j] = ruleset[7 - (4 * left + 2 * centre + right)]

		gen = next_gen[:]

	scene.fill((20, 20, 20))
	pg.display.update()
	sleep(0.5)
	for i in range(IMG_SIZE):
		for j in range(IMG_SIZE):
			c = (255, 255, 255) if plot[i][j] == 1 else (20, 20, 20)
			pg.draw.rect(scene, c, pg.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
		pg.display.update()
		sleep(1e-4)

		for event in pg.event.get():
			if event.type == pg.QUIT:
				sys.exit()


if __name__ == '__main__':
	pg.init()
	scene = pg.display.set_mode((IMG_SIZE * CELL_SIZE, IMG_SIZE * CELL_SIZE))

	rules = [18, 30, 45, 57, 105, 161]  # Interesting rules
	i = 0

	while True:
		ruleset = get_ruleset(rules[i])
		pg.display.set_caption(f'Elementary Cellular Automata (rule {rules[i]}: {ruleset})')
		generate_plot(ruleset)
		i = (i + 1) % len(rules)
		sleep(1.5)
