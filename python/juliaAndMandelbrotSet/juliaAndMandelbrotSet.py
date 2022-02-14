# Julia/Mandelbrot Set Generator
# Author: Sam Barba
# Created 22/09/2021

# Click: Select point to set as origin (0,0)
# C: Centre image around origin (0,0)
# Num keys 2,4,8,0: Magnify around origin by 2/4/8/100 times, respectively
# T: Toggle axes
# R: Reset

from math import log
import pygame as pg
import sys

WIDTH = 750
HEIGHT = 500
MAX_ITERATIONS = 200
RGB_PALETTE = [(0, 20, 100), (30, 100, 200), (230, 255, 255), (255, 170, 0)]

# Set to true if drawing Mandelbrot set
mandelbrotSet = True

# Or, change these a,b values: A complex number c is defined as c = a + bi.
# For a given c, the Julia set is the set of all complex z for which the iteration z = z^2 + c does not diverge.
# For almost all values of c, these sets are fractals.
# Interesting a,b pairs:
# a, b = -0.79, 0.15
# a, b = -0.75, 0.11
# a, b = -0.7, 0.38
# a, b = -0.4, 0.6
# a, b = 0, 0.8
a, b = 0.28, 0.008

scale = 200
xAxis = xOffset = WIDTH / 2
yAxis = yOffset = HEIGHT / 2
showAxes = True

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def draw():
	global a, b

	for x in range(WIDTH):
		for y in range(HEIGHT):
			# Real and imaginary parts of z
			za = (x - xOffset) / scale
			zb = (y - yOffset) / scale
			if mandelbrotSet:
				a, b = za, zb

			i = 0
			aa = bb = 0

			# Test, as we iterate z = z^2 + c, does z diverge?
			# Let infinity be 16
			while aa + bb <= 4 and i < MAX_ITERATIONS:
				aa = za * za
				bb = zb * zb
				twoAB = 2 * za * zb
				za = aa - bb + a
				zb = twoAB + b
				i += 1

			if i < MAX_ITERATIONS:
				# Apply smooth colouring
				logZ = log(aa + bb) / 2
				n = log(logZ / log(2)) / log(2)
				i += 1 - n

				idx = i / MAX_ITERATIONS * len(RGB_PALETTE)
				colour1 = RGB_PALETTE[int(idx) % len(RGB_PALETTE)]
				colour2 = RGB_PALETTE[int(idx + 1) % len(RGB_PALETTE)]
				colour = linearInterpolate(colour1, colour2, idx % 1) # Mod 1 to get fractional part
				scene.set_at((x, y), colour)
			else:
				scene.set_at((x, y), (0, 0, 0))

	if showAxes:
		pg.draw.line(scene, (255, 255, 255), (0, round(yAxis)), (WIDTH, round(yAxis)))
		pg.draw.line(scene, (255, 255, 255), (round(xAxis), 0), (round(xAxis), HEIGHT))

	pg.display.flip()

def linearInterpolate(colour1, colour2, t):
	return tuple(round(c1 + t * (c2 - c1)) for c1, c2 in zip(colour1, colour2))

def centreAroundOrigin():
	global xAxis, yAxis, xOffset, yOffset

	xOffset -= (xAxis - WIDTH / 2)
	yOffset -= (yAxis - HEIGHT / 2)
	xAxis = WIDTH / 2
	yAxis = HEIGHT / 2

def magnify(factor):
	global scale, xOffset, yOffset

	scale *= factor
	xOffset = factor * (xOffset - xAxis) + xAxis
	yOffset = factor * (yOffset - yAxis) + yAxis

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

pg.init()
pg.display.set_caption("Julia/Mandelbrot Set")
scene = pg.display.set_mode((WIDTH, HEIGHT))

draw()

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)

		elif event.type == pg.MOUSEBUTTONDOWN:
			if event.button == 1: # Left-click
				print("Setting origin... ", end="")
				xAxis, yAxis = event.pos
				draw()
				print("Done")

		elif event.type == pg.KEYDOWN:
			if event.key == pg.K_c: # Centre image around origin
				print("Centering image around origin... ", end="")
				centreAroundOrigin()
				draw()
				print("Done")

			elif event.key in (pg.K_2, pg.K_4, pg.K_8, pg.K_0): # Magnify
				if event.key == pg.K_0:
					print("Magnifying by 100... ", end="")
					magnify(100)
				else:
					# Subtract 48 to get magnification factor ('2' key id = 50)
					factor = event.key - 48
					print(f"Magnifying by {factor}... ", end="")
					magnify(factor)

				centreAroundOrigin()
				draw()
				print("Done")

			elif event.key == pg.K_t: # Toggle axes
				print("Toggling axes... ", end="")
				showAxes = not showAxes
				draw()
				print("Done")

			elif event.key == pg.K_r: # Reset
				print("Resetting... ", end="")
				scale = 200
				xAxis = xOffset = WIDTH / 2
				yAxis = yOffset = HEIGHT / 2
				showAxes = True
				draw()
				print("Done")
