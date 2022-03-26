# Drawing Ulam's spiral (of primes)
# Author: Sam Barba
# Created 26/03/2022

import pygame as pg
import sys

# Ensure this is odd
GRID_SIZE = 899

# Don't change these
GRID_SIZE = min(GRID_SIZE, 899)
CELL_SIZE = 900 // GRID_SIZE

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def is_prime(n):
	if n in (2, 3): return True
	if n < 2 or n % 2 == 0: return False
	if n < 9: return True
	if n % 3 == 0: return False

	lim = int(n ** 0.5) + 1
	for i in range(5, lim, 6):
		if n % i == 0 or n % (i + 2) == 0:
			return False

	return True

def draw():
	x = y = (GRID_SIZE * CELL_SIZE) // 2
	state = 0
	num_steps = 1
	turn_counter = 1
	font = pg.font.SysFont("consolas", 14)

	for n in range(1, GRID_SIZE ** 2 + 1):
		colour = (255, 0, 0) if is_prime(n) else (0, 0, 0)
		pg.draw.rect(scene, colour, pg.Rect(x - CELL_SIZE / 2, y - CELL_SIZE / 2, CELL_SIZE, CELL_SIZE))

		# Only enough room in a cell to draw numbers up to 1000 (31^2 = 961, 33^2 = 1089)
		if GRID_SIZE <= 31:
			num_lbl = font.render(str(n), True, (220, 220, 220))
			lbl_rect = num_lbl.get_rect(center=(x, y))
			scene.blit(num_lbl, lbl_rect)

		if state == 0: x += CELL_SIZE
		elif state == 1: y -= CELL_SIZE
		elif state == 2: x -= CELL_SIZE
		elif state == 3: y += CELL_SIZE

		if n % num_steps == 0:
			state = (state + 1) % 4
			turn_counter += 1
			if turn_counter % 2 == 0:
				num_steps += 1

	pg.display.update()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
	pg.init()
	pg.display.set_caption("Drawing Ulam's spiral")
	scene = pg.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))

	# Test prime function (1000th prime should be 7919)
	count = n = 0
	while count < 1000:
		if is_prime(n):
			count += 1
		n += 1
	print("1000th prime:", n - 1)

	draw()

	while True:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				sys.exit(0)
