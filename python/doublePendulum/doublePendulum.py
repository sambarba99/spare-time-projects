"""
Simulation of a double pendulum

Author: Sam Barba
Created 22/09/2021

Controls:
R: reset
"""

from math import pi, sin, cos
import pygame as pg
import random
import sys

R1 = 300
R2 = 300
M1 = 10
M2 = 10
G = 0.1
WIDTH, HEIGHT = 1300, 800
FPS = 120

a1 = a2 = vel1 = vel2 = 0
positions = []

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def draw():
	global a1, a2, vel1, vel2, positions

	scene.fill((20, 20, 20))

	num1 = -G * (2 * M1 + M2) * sin(a1)
	num2 = -M2 * G * sin(a1 - 2 * a2)
	num3 = -2 * sin(a1 - a2) * M2
	num4 = vel2 * vel2 * R2 + vel1 * vel1 * R1 * cos(a1 - a2)
	den = R1 * (2 * M1 + M2 - M2 * cos(2 * a1 - 2 * a2))
	a1acc = (num1 + num2 + num3 * num4) / den

	num1 = 2 * sin(a1 - a2)
	num2 = vel1 * vel1 * R1 * (M1 + M2)
	num3 = G * (M1 + M2) * cos(a1)
	num4 = vel2 * vel2 * R2 * M2 * cos(a1 - a2)
	den = R2 * (2 * M1 + M2 - M2 * cos(2 * a1 - 2 * a2))
	a2acc = num1 * (num2 + num3 + num4) / den

	x1 = R1 * sin(a1) + WIDTH / 2
	y1 = R1 * cos(a1)

	x2 = x1 + R2 * sin(a2)
	y2 = y1 + R2 * cos(a2)

	positions.append([round(x2), round(y2)])
	positions = positions[-300:]

	pg.draw.line(scene, (220, 220, 220), (WIDTH / 2, 0), (x1, y1))
	pg.draw.line(scene, (220, 220, 220), (x1, y1), (x2, y2))
	pg.draw.circle(scene, (220, 220, 220), (x1, y1), 10)
	pg.draw.circle(scene, (220, 220, 220), (x2, y2), 10)

	if len(positions) > 1:
		for i in range(len(positions) - 1):
			draw_line(*positions[i], *positions[i + 1])

	pg.display.update()

	vel1 += a1acc
	vel2 += a2acc
	a1 += vel1
	a2 += vel2

	# Damping
	vel1 *= 0.9999
	vel2 *= 0.9999

# Bresenham's algorithm
def draw_line(x1, y1, x2, y2):
	dx = abs(x2 - x1)
	dy = -abs(y2 - y1)
	sx = 1 if x1 < x2 else -1
	sy = 1 if y1 < y2 else -1
	err = dx + dy

	while True:
		scene.set_at((x1, y1), (230, 20, 20))

		if (x1, y1) == (x2, y2): return

		e2 = 2 * err
		if e2 >= dy:
			err += dy
			x1 += sx
		if e2 <= dx:
			err += dx
			y1 += sy

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	a1 = random.uniform(0, 2 * pi)
	a2 = random.uniform(0, 2 * pi)

	pg.init()
	pg.display.set_caption('Double Pendulum')
	scene = pg.display.set_mode((WIDTH, HEIGHT))
	clock = pg.time.Clock()

	while True:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				sys.exit(0)
			elif event.type == pg.KEYDOWN:
				if event.key == pg.K_r:
					a1 = random.uniform(0, 2 * pi)
					a2 = random.uniform(0, 2 * pi)
					vel1 = vel2 = 0
					positions = []

		draw()
		clock.tick(FPS)
