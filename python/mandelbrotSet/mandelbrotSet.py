# Mandelbrot Set
# Author: Sam Barba
# Created 22/09/2021

# C: centre axes
# T: toggle axes
# R: reset
# Click/scroll: select point and zoom in/out on it

import pygame as pg
import sys

WIDTH = 750
HEIGHT = 500
ZOOM = 175
SCALE_MULT_FACTOR = 3

scale = 1
xAxis = WIDTH / 2
yAxis = HEIGHT / 2
xOffset = xAxis
yOffset = yAxis
showAxes = True

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def draw(scene):
	for x in range(WIDTH):
		for y in range(HEIGHT):
			a = b = 0
			p = (x - xOffset) / (ZOOM * scale)
			q = (y - yOffset) / (ZOOM * scale)

			# Determine whether a point is in the set
			n = 0
			while a * a + b * b <= 16 and n < 25:
				realZ = a * a - b * b + p
				b = 2 * a * b + q
				a = realZ
				n += 1

			c = mapRange(n, 0, 25, 0, 255)
			scene.set_at((x, y), (c, c, c))

	if showAxes:
		for re in range(WIDTH):
			scene.set_at((re, round(yAxis)), (255, 0, 0))
		for im in range(HEIGHT):
			scene.set_at((round(xAxis), im), (255, 0, 0))

	pg.display.flip()

def mapRange(x, fromLo, fromHi, toLo, toHi):
	return (x - fromLo) * (toHi - toLo) / (fromHi - fromLo) + toLo

def scaleDrawing(eventButton):
	global scale, xOffset, yOffset

	if eventButton == 4: # Scroll up
		scale *= SCALE_MULT_FACTOR
		xOffset *= SCALE_MULT_FACTOR
		yOffset *= SCALE_MULT_FACTOR
		afterZoomX = xAxis * SCALE_MULT_FACTOR
		afterZoomY = yAxis * SCALE_MULT_FACTOR
	else: # eventButton == 5, so scrolling down
		scale /= SCALE_MULT_FACTOR
		xOffset /= SCALE_MULT_FACTOR
		yOffset /= SCALE_MULT_FACTOR
		afterZoomX = xAxis / SCALE_MULT_FACTOR
		afterZoomY = yAxis / SCALE_MULT_FACTOR

	xOffset -= afterZoomX - xAxis
	yOffset -= afterZoomY - yAxis

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

pg.init()
pg.display.set_caption("Mandelbrot Set")
scene = pg.display.set_mode((WIDTH, HEIGHT))

draw(scene)

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)
		elif event.type == pg.KEYDOWN:
			if event.key == pg.K_c: # Centre axes
				xOffset -= (xAxis - WIDTH / 2)
				yOffset -= (yAxis - HEIGHT / 2)
				xAxis = WIDTH / 2
				yAxis = HEIGHT / 2
				draw(scene)

			elif event.key == pg.K_t: # Toggle axes
				showAxes = not showAxes
				draw(scene)

			elif event.key == pg.K_r: # Reset
				scale = 1
				xAxis = xOffset = WIDTH / 2
				yAxis = yOffset = HEIGHT / 2
				showAxes = True
				draw(scene)

		elif event.type == pg.MOUSEBUTTONDOWN:
			if event.button == 1: # Left click
				xAxis, yAxis = event.pos
				draw(scene)
			elif event.button in [4, 5]: # Scrolling up/down
				scaleDrawing(event.button)
				draw(scene)
