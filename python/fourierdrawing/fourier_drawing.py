"""
Drawing with the Discrete Fourier Transform

Author: Sam Barba
Created 22/09/2022

Controls:
Right-click: enter/exit drawing mode
Left-click [and drag]: draw freestyle
H/P: draw preset heart or pi symbol
Z/X/C: Use 3/8/20 epicycles to draw
V: Use all calculated epicycles to draw
Space: toggle animation
"""

import numpy as np
import pygame as pg
import sys

from presets import HEART, PI

SIZE = 800
FPS = 120

scene = None
num_epicycles = -1  # No limit for how many epicycles to draw with

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def compute_fourier_from_coords(drawing_coords, coords_name):
	print(f'\nComputing DFT for {coords_name}... ', end='')

	# Normalise around origin (0,0)
	normalised_coords = [(x - SIZE / 2, y - SIZE / 2) for x, y in drawing_coords]

	# Fill any gaps
	interpolated = interpolate(normalised_coords)

	# Convert to complex
	complex_ = [xi + yi * 1j for xi, yi in interpolated]

	fourier = dft(complex_)

	print('done')

	return fourier

def interpolate(coords):
	def bresenham(x1, y1, x2, y2):
		"""Bresenham's algorithm"""

		dx = abs(x2 - x1)
		dy = -abs(y2 - y1)
		sx = 1 if x1 < x2 else -1
		sy = 1 if y1 < y2 else -1
		err = dx + dy
		result = []

		while True:
			result.append((x1, y1))

			if (x1, y1) == (x2, y2): return result

			e2 = 2 * err
			if e2 >= dy:
				err += dy
				x1 += sx
			if e2 <= dx:
				err += dx
				y1 += sy

	# Remove duplicate coords but maintain order
	coords = sorted(set(coords), key=coords.index)

	if len(coords) < 2: return coords

	interpolated_coords = []
	for coord, next_coord in zip(coords[:-1], coords[1:]):
		bres_coords = bresenham(*coord, *next_coord)
		interpolated_coords.extend(bres_coords)

	return interpolated_coords

def dft(x):
	"""
	Discrete Fourier Transform (see https://en.wikipedia.org/wiki/Discrete_Fourier_transform,
	section `Definition`)
	"""

	N = len(x)
	X = [None] * N

	for k in range(N):
		sum_ = sum(xn * np.exp(-1j * 2 * np.pi * k * n / N) for n, xn in enumerate(x))
		sum_ /= N  # Average the sum's contribution over N

		X[k] = {'re': sum_.real,
			'im': sum_.imag,
			'frequency': k,
			'amplitude': abs(sum_),
			'phase': np.arctan2(sum_.imag, sum_.real)}

	# Descending order of amplitude
	return sorted(X, key=lambda item: -item['amplitude'])

def epicycles(x, y, fourier, time):
	global num_epicycles

	lim = len(fourier) if num_epicycles == -1 else min(num_epicycles, len(fourier))

	for f in fourier[:lim]:
		prev_x, prev_y = x, y
		freq = f['frequency']
		radius = f['amplitude']
		phase = f['phase']
		x += radius * np.cos(freq * time + phase)
		y += radius * np.sin(freq * time + phase)

		pg.draw.circle(scene, (80, 80, 80), (prev_x, prev_y), radius, 1)
		pg.draw.line(scene, (255, 255, 255), (prev_x, prev_y), (x, y))

	return x, y

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	global scene, num_epicycles

	pg.init()
	pg.display.set_caption('Drawing with the Discrete Fourier Transform')
	scene = pg.display.set_mode((SIZE, SIZE))
	clock = pg.time.Clock()

	left_btn_down = False
	user_drawing_mode = True
	user_drawing_coords = []
	fourier = None
	paused = False
	path = []
	time = dt = 0

	# For changing no. epicycles to draw with
	epicycle_dict = {pg.K_z: 3, pg.K_x: 8, pg.K_c: 20, pg.K_v: -1}

	print('\nDraw something or select a preset\n(right-click to exit drawing mode)')

	while True:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				sys.exit(0)

			elif event.type == pg.MOUSEBUTTONDOWN:
				if event.button == 1:  # Left-click
					if user_drawing_mode:
						left_btn_down = True
				elif event.button == 3:  # Right-click
					user_drawing_mode = not user_drawing_mode
					if user_drawing_mode:  # Start drawing
						print('\nDraw something or select a preset\n(right-click to exit drawing mode)')
						user_drawing_coords = []  # Clear for new drawing
						path = []  # Clear any previously rendered path
						time = 0
					else:  # Finished drawing
						if len(user_drawing_coords) < 2:
							print('\nNeed at least 2 coordinates')
							user_drawing_mode = True
						else:
							fourier = compute_fourier_from_coords(user_drawing_coords, 'user drawing')
							dt = 2 * np.pi / len(fourier)

			elif event.type == pg.MOUSEMOTION and left_btn_down and user_drawing_mode:
				user_drawing_coords.append(event.pos)

			elif event.type == pg.MOUSEBUTTONUP and left_btn_down and user_drawing_mode:
				user_drawing_coords.append(event.pos)
				left_btn_down = False

			elif event.type == pg.KEYDOWN:
				if event.key in epicycle_dict:
					num_epicycles = epicycle_dict[event.key]
					if num_epicycles == -1:
						print(f'\nDrawing with all epicycles ({len(fourier)})')
					else:
						print(f'\nDrawing with {num_epicycles} epicycles')
					path = []
					time = 0
				elif event.key in (pg.K_h, pg.K_p):
					if event.key == pg.K_h:
						fourier = compute_fourier_from_coords(HEART, 'heart')
					else:
						fourier = compute_fourier_from_coords(PI, 'pi symbol')
					user_drawing_mode = paused = False
					path = []
					time = 0
					dt = 2 * np.pi / len(fourier)
				elif event.key == pg.K_SPACE:
					paused = not paused

		if paused: continue

		scene.fill((0, 0, 0))

		if user_drawing_mode:
			for point in user_drawing_coords:
				scene.set_at(point, (255, 0, 0))
		else:  # Draw Fourier
			epicycle_final_pos = epicycles(SIZE / 2, SIZE / 2, fourier, time)
			path.append(list(map(round, epicycle_final_pos)))
			for point, next_point in zip(path[:-1], path[1:]):
				pg.draw.line(scene, (255, 0, 0), point, next_point)

			time += dt
			if time > 2 * np.pi:  # Done a full revolution, so reset
				path = []
				time = 0

		pg.display.update()
		clock.tick(FPS)

if __name__ == '__main__':
	main()
