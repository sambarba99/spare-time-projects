"""
Julia/Mandelbrot Set Generator

Author: Sam Barba
Created 22/09/2021

Controls:
Click: select point to set as origin (0,0)
Num keys 2,4,8,0: magnify around origin by 2/4/8/100 times, respectively
T: toggle axes
R: reset
"""

from math import log
import sys

import pygame as pg

WIDTH, HEIGHT = 800, 550
MAX_ITERATIONS = 200
RGB_PALETTE = [(0, 20, 100), (30, 100, 200), (230, 255, 255), (255, 170, 0)]
LOG2 = log(2)

# Set to true if drawing Mandelbrot set...
mandelbrot_set = True

# ...or, change this complex number c: for a given c, its Julia set is the set of all complex z for
# which the iteration z = z^2 + c doesn't diverge. For almost all c, these sets are fractals.
# Interesting values of c:
# c = -0.79 + 0.15j
# c = -0.75 + 0.11j
# c = -0.7 + 0.38j
# c = -0.4 + 0.6j
# c = 0.8j
c = 0.28 + 0.008j

scale = 200
x_axis = x_offset = WIDTH / 2
y_axis = y_offset = HEIGHT / 2
show_axes = True
scene = None

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def draw():
	def linear_interpolate(colour1, colour2, t):
		return tuple(round(c1 + t * (c2 - c1)) for c1, c2 in zip(colour1, colour2))

	global c

	scene.fill((0, 0, 0))

	for y in range(HEIGHT):
		for x in range(WIDTH):
			z_real = (x - x_offset) / scale
			z_imag = (y - y_offset) / scale
			z = z_real + z_imag * 1j
			if mandelbrot_set:
				c = z

			# Test, as we iterate z = z^2 + c, does z diverge?
			i = 0
			while abs(z) < 2 and i < MAX_ITERATIONS:
				z = z ** 2 + c
				i += 1

			if i < MAX_ITERATIONS:
				# Apply smooth colouring
				log_ = log(abs(z)) / 2
				n = log(log_ / LOG2) / LOG2
				i += 1 - n
				idx = i / MAX_ITERATIONS * len(RGB_PALETTE)
				colour1 = RGB_PALETTE[int(idx) % len(RGB_PALETTE)]
				colour2 = RGB_PALETTE[int(idx + 1) % len(RGB_PALETTE)]
				colour = linear_interpolate(colour1, colour2, idx % 1)  # Mod 1 to get fractional part
				scene.set_at((x, y), colour)

	if show_axes:
		pg.draw.line(scene, (255, 255, 255), (0, round(y_axis)), (WIDTH, round(y_axis)))
		pg.draw.line(scene, (255, 255, 255), (round(x_axis), 0), (round(x_axis), HEIGHT))

	pg.display.update()

def centre_around_origin():
	global x_axis, y_axis, x_offset, y_offset

	x_offset -= (x_axis - WIDTH / 2)
	y_offset -= (y_axis - HEIGHT / 2)
	x_axis = WIDTH / 2
	y_axis = HEIGHT / 2

def magnify(factor):
	global scale, x_offset, y_offset

	scale *= factor
	x_offset = factor * (x_offset - x_axis) + x_axis
	y_offset = factor * (y_offset - y_axis) + y_axis

def calculate_origin():
	orig_x = (x_axis - x_offset) / scale
	orig_y = -(y_axis - y_offset) / scale

	return f'({orig_x}, {orig_y})'

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	global scale, x_axis, x_offset, y_axis, y_offset, show_axes, scene

	pg.init()
	pg.display.set_caption('Julia/Mandelbrot Set')
	scene = pg.display.set_mode((WIDTH, HEIGHT))

	draw()

	while True:
		for event in pg.event.get():
			match event.type:
				case pg.QUIT: sys.exit()
				case pg.MOUSEBUTTONDOWN:
					if event.button == 1:  # Left-click
						print('Setting origin... ', end='')
						x_axis, y_axis = event.pos
						centre_around_origin()
						draw()
						print(f'origin = {calculate_origin()}')
				case pg.KEYDOWN:
					match event.key:
						case pg.K_2 | pg.K_4 | pg.K_8 | pg.K_0:  # Magnify
							if event.key == pg.K_0:
								factor = 100
							else:
								# Subtract 48 to get magnification factor ('2' key id = 50)
								factor = event.key - 48

							print(f'Magnifying by {factor}... ', end='')
							magnify(factor)
							draw()
							print('Done')
						case pg.K_t:  # Toggle axes
							print('Toggling axes... ', end='')
							show_axes = not show_axes
							draw()
							print('Done')
						case pg.K_r:  # Reset
							print('Resetting... ', end='')
							scale = 200
							x_axis = x_offset = WIDTH / 2
							y_axis = y_offset = HEIGHT / 2
							show_axes = True
							draw()
							print('Done')

if __name__ == '__main__':
	main()
