# Python drawing
# Author: Sam Barba
# Created 29/10/2018

from pen import Pen
import pygame as pg
import random
import sys

WIDTH = 1500.0
HEIGHT = 900.0

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def cantor(x=WIDTH * 0.05, y=HEIGHT / 3, l=WIDTH * 0.9):
	if l > 1:
		pen.go_to(x, y)
		pen.move(l)
		cantor(x, y + 40, l / 3)
		cantor(x + l * 2 / 3, y + 40, l / 3)

def dragon(lvl=14, size=4, theta=45):
	if lvl:
		pen.turn(theta)
		dragon(lvl - 1, size)
		pen.turn(-theta * 2)
		dragon(lvl - 1, size, -45)
		pen.turn(theta)
	else:
		pen.move(size)

def sierpinski_triangle(size=HEIGHT, lvl=7):
	if lvl == 0:
		for i in range(3):
			pen.move(size)
			pen.turn(-120)
	else:
		sierpinski_triangle(size / 2, lvl - 1)
		pen.move(size / 2)
		sierpinski_triangle(size / 2, lvl - 1)
		pen.move(-size / 2)
		pen.turn(-60)
		pen.move(size / 2)
		pen.turn(60)
		sierpinski_triangle(size / 2, lvl - 1)
		pen.turn(-60)
		pen.move(-size / 2)
		pen.turn(60)

def koch_snowflake(size=WIDTH * 0.95, lvl=6):
	if lvl:
		for angle in [-60, 120, -60, 0]:
			koch_snowflake(size / 3, lvl - 1)
			pen.turn(angle)
	else:
		pen.move(size)

def t_square(x=WIDTH / 2, y=HEIGHT / 2, size=HEIGHT * 0.45):
	if size < 4: return

	pen.go_to(x - size / 2, y - size / 2)
	for _ in range(4):
		pen.move(size)
		pen.turn(90)

	t_square(x - size / 2, y - size / 2, size / 2)
	t_square(x - size / 2, y + size / 2, size / 2)
	t_square(x + size / 2, y - size / 2, size / 2)
	t_square(x + size / 2, y + size / 2, size / 2)

def tree(angle, size=HEIGHT * 0.2):
	if size > 10:
		pen.move(size)
		pen.turn(angle)
		tree(angle, size * 0.75)
		pen.turn(-angle * 2)
		tree(angle, size * 0.75)
		pen.turn(angle)
		pen.move(-size)

def spiral(angle):
	for i in range(300):
		pen.move(i)
		pen.turn(-angle)

def hilbert(size=5, level=7, angle=90):
	if level:
		pen.turn(angle)
		hilbert(size, level - 1, -angle)
		pen.move(size)
		pen.turn(-angle)
		hilbert(size, level - 1, angle)
		pen.move(size)
		hilbert(size, level - 1, angle)
		pen.turn(-angle)
		pen.move(size)
		hilbert(size, level - 1, -angle)
		pen.turn(angle)

def rand_walk(step=2):
	path = [pen.pos()]

	while True:
		pen.turn(random.choice([0, 90, 180, 270]))
		pen.move(step, draw=False)
		path.append(pen.pos())

		if not 0 <= pen.x < WIDTH or not 0 <= pen.y < HEIGHT:
			break

	for idx, point in enumerate(path[:-1]):
		start_x, start_y = point
		end_x, end_y = path[idx + 1]
		c = round(map_range(idx, 0, len(path), 30, 255))
		pg.draw.line(scene, (c, c, c), (start_x, start_y), (end_x, end_y))

	pg.display.flip()

def map_range(x, from_lo, from_hi, to_lo, to_hi):
	return (x - from_lo) * (to_hi - to_lo) / (from_hi - from_lo) + to_lo

def reset(x=WIDTH / 2, y=HEIGHT/2, heading=-90):
	scene.fill((0, 0, 0))
	pen.go_to(x, y)
	pen.heading = heading
	wait_for_click()

def wait_for_click():
	while True:
		for event in pg.event.get():
			if event.type == pg.MOUSEBUTTONDOWN:
				return
			elif event.type == pg.QUIT:
				pg.quit()
				sys.exit(0)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

pg.init()
pg.display.set_caption("Python Drawing")
scene = pg.display.set_mode((int(WIDTH), int(HEIGHT)))

pen = Pen(scene, WIDTH / 2, HEIGHT / 2, 0)

wait_for_click()
cantor()
reset(WIDTH * 0.35, HEIGHT * 0.35, 0)

dragon()
reset(WIDTH * 0.22, HEIGHT * 0.93, 0)

sierpinski_triangle()
reset(WIDTH * 0.03, HEIGHT * 0.65, 0)

koch_snowflake()
reset(heading=0)

t_square()
for angle in [10, 20, 30]:
	reset(y=HEIGHT * 0.9)
	tree(angle)

for angle in [45, 51, 60, 72, 103, 120, 144, 168]:
	reset()
	spiral(angle)
reset(WIDTH * 0.71, HEIGHT * 0.84, 180)

hilbert()
reset()

rand_walk()
wait_for_click()
