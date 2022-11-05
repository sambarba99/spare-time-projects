"""
Double pendulum simulation

Author: Sam Barba
Created 22/09/2021

Controls:
R: reset
Space: play/pause
"""

import sys

import numpy as np
import pygame as pg

R1 = 300
R2 = 300
M1 = 10
M2 = 10
G = 0.1
WIDTH, HEIGHT = 1300, 800
FPS = 120
COLOUR_DECAY = 0.999

a1 = a2 = vel1 = vel2 = 0
positions = []

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def draw():
	def draw_line(x1, y1, x2, y2, colour):
		"""Bresenham's algorithm"""

		dx = abs(x2 - x1)
		dy = -abs(y2 - y1)
		sx = 1 if x1 < x2 else -1
		sy = 1 if y1 < y2 else -1
		err = dx + dy

		while True:
			scene.set_at((x1, y1), colour)

			if (x1, y1) == (x2, y2): return

			e2 = 2 * err
			if e2 >= dy:
				err += dy
				x1 += sx
			if e2 <= dx:
				err += dx
				y1 += sy

	global a1, a2, vel1, vel2, positions

	scene.fill((0, 0, 0))

	num1 = -G * (2 * M1 + M2) * np.sin(a1)
	num2 = -M2 * G * np.sin(a1 - 2 * a2)
	num3 = -2 * np.sin(a1 - a2) * M2
	num4 = vel2 * vel2 * R2 + vel1 * vel1 * R1 * np.cos(a1 - a2)
	den = R1 * (2 * M1 + M2 - M2 * np.cos(2 * a1 - 2 * a2))
	a1acc = (num1 + num2 + num3 * num4) / den

	num1 = 2 * np.sin(a1 - a2)
	num2 = vel1 * vel1 * R1 * (M1 + M2)
	num3 = G * (M1 + M2) * np.cos(a1)
	num4 = vel2 * vel2 * R2 * M2 * np.cos(a1 - a2)
	den = R2 * (2 * M1 + M2 - M2 * np.cos(2 * a1 - 2 * a2))
	a2acc = num1 * (num2 + num3 + num4) / den

	x1 = R1 * np.sin(a1) + WIDTH / 2
	y1 = R1 * np.cos(a1)

	x2 = x1 + R2 * np.sin(a2)
	y2 = y1 + R2 * np.cos(a2)

	pg.draw.line(scene, (220, 220, 220), (WIDTH / 2, 0), (x1, y1))
	pg.draw.line(scene, (220, 220, 220), (x1, y1), (x2, y2))
	pg.draw.circle(scene, (220, 220, 220), (x1, y1), 10)
	pg.draw.circle(scene, (220, 220, 220), (x2, y2), 10)

	positions.append([round(x2), round(y2)])

	if len(positions) > 1:
		l = len(positions)
		for i in range(l - 1):
			red = 255 * COLOUR_DECAY ** (l - i)
			draw_line(*positions[i], *positions[i + 1], (int(red), 0, 0))

	if len(positions) > 2000: positions.pop(0)

	pg.display.update()

	vel1 += a1acc
	vel2 += a2acc
	a1 += vel1
	a2 += vel2

	# Damping
	vel1 *= 0.9999
	vel2 *= 0.9999

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	a1, a2 = np.random.uniform(0, 2 * np.pi, size=2)

	pg.init()
	pg.display.set_caption('Double Pendulum')
	scene = pg.display.set_mode((WIDTH, HEIGHT))
	clock = pg.time.Clock()
	paused = False

	while True:
		for event in pg.event.get():
			match event.type:
				case pg.QUIT: sys.exit()
				case pg.KEYDOWN:
					match event.key:
						case pg.K_r:
							a1, a2 = np.random.uniform(0, 2 * np.pi, size=2)
							vel1 = vel2 = 0
							positions = []
							paused = False
						case pg.K_SPACE:
							paused = not paused

		if not paused:
			draw()
			clock.tick(FPS)
