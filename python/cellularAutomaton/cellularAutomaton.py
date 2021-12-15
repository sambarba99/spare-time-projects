# Elementary Cellular Automaton demo
# Author: Sam Barba
# Created 23/09/2021

import pygame as pg
import sys
from time import sleep

IMG_SIZE = 299 # Rows and columns
CELL_SIZE = 3  # Size of each cell

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def setRuleSet(n):
	if n < 1 or n > 255:
		raise ValueError("Rule no. must be 1 - 255")

	ruleSet = [0] * 8
	i = 7
	while n:
		ruleSet[i] = n % 2
		n //= 2
		i -= 1

	return ruleSet

def generatePlot(scene, ruleSet):
	gen = [0] * IMG_SIZE
	gen[IMG_SIZE // 2] = 1 # Turn on centre pixel of first generation
	plot = [[0] * IMG_SIZE] * IMG_SIZE

	for x in range(IMG_SIZE):
		plot[x] = gen[:]
		nextGen = [0] * IMG_SIZE

		for y in range(IMG_SIZE):
			left = 0 if y == 0 else gen[y - 1]
			centre = gen[y]
			right = 0 if y == IMG_SIZE - 1 else gen[y + 1]
			nextGen[y] = ruleSet[7 - (4 * left + 2 * centre + right)]

		gen = nextGen[:]

	for x in range(IMG_SIZE):
		for y in range(IMG_SIZE):
			c = (220, 220, 220) if plot[y][x] == 1 else (20, 20, 20)
			pg.draw.rect(scene, c, pg.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

	pg.display.flip()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

pg.init()
pg.display.set_caption("Elementary Cellular Automaton")
scene = pg.display.set_mode((IMG_SIZE * CELL_SIZE, IMG_SIZE * CELL_SIZE))

rules = [18, 30, 45, 54, 57, 60, 73, 105, 129, 137, 151, 153, 161, 165] # Interesting rules
i = 0

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)

	ruleSet = setRuleSet(rules[i])
	generatePlot(scene, ruleSet)
	print(f"Rule {rules[i]}: {ruleSet}")
	i = (i + 1) % len(rules)
	sleep(2)
