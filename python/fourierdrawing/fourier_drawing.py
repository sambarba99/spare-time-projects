"""
Drawing with the Discrete Fourier Transform

Author: Sam Barba
Created 22/09/2022

Controls:
Right-click: enter/exit drawing mode
Left-click [and drag]: draw freestyle
P/G/T/C: draw preset pi symbol/guitar/T. Rex/Colosseum
Up/down arrow: increase/decrease number of epicycles to draw with
Space: toggle animation
"""

import json
import numpy as np
import pygame as pg
import sys

from presets import PI, GUITAR, T_REX, COLOSSEUM

SIZE = 800
FPS = 60

scene = None
n_epicycles = 0

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def load_fourier_from_file(path, image_name):
	"""For presets, just load saved DFTs from their file intead of computing again"""

	print(f'\nLoading saved DFT for {image_name}... ', end='')

	with open(path, 'r') as file:
		data = file.read()[:-1].split('\n')

	fourier = [json.loads(line.replace("'", '"')) for line in data]

	print(f'done ({len(fourier)} total epicycles)')

	return fourier

def compute_fourier_from_coords(drawing_coords, image_name):
	print(f'\nComputing DFT for {image_name}... ', end='')

	# Normalise around origin (0,0)
	normalised_coords = [(x - SIZE / 2, y - SIZE / 2) for x, y in drawing_coords]

	# Fill any gaps
	interpolated = interpolate(normalised_coords)

	# Skip 3 path points at a time (don't need that much resolution)
	drawing_path = interpolated[::4]

	# Convert to complex
	complex_ = [x + y * 1j for x, y in drawing_path]

	fourier = dft(complex_)

	print(f'done ({len(fourier)} total epicycles)')

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

	if len(coords) < 2: return coords

	interpolated_coords = []
	for coord, next_coord in zip(coords[:-1], coords[1:]):
		bres_coords = bresenham(*coord, *next_coord)
		interpolated_coords.extend(bres_coords)

	return interpolated_coords

def dft(x):
	"""
	Discrete Fourier Transform (see https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Definition)
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
	for f in fourier[:n_epicycles]:
		prev_x, prev_y = x, y
		freq = f['frequency']
		radius = f['amplitude']
		phase = f['phase']
		x += radius * np.cos(freq * time + phase)
		y += radius * np.sin(freq * time + phase)

		pg.draw.circle(scene, (60, 60, 60), (prev_x, prev_y), radius, 1)
		pg.draw.line(scene, (200, 200, 200), (prev_x, prev_y), (x, y))

	return x, y

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	global scene, n_epicycles

	pg.init()
	pg.display.set_caption('Drawing with the Discrete Fourier Transform')
	scene = pg.display.set_mode((SIZE, SIZE))
	clock = pg.time.Clock()

	left_btn_down = False
	user_drawing_mode = True
	user_drawing_coords = []
	fourier = None
	path = []
	time = dt = 0
	paused = False

	print('\nDraw something or select a preset\n(right-click to exit drawing mode)')

	while True:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				sys.exit()

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
							print('\nNeed at least 2 points')
							user_drawing_mode = True
						else:
							fourier = compute_fourier_from_coords(user_drawing_coords, 'user drawing')
							dt = 2 * np.pi / len(fourier)
							n_epicycles = len(fourier)
							paused = False

			elif event.type == pg.MOUSEMOTION and left_btn_down and user_drawing_mode:
				user_drawing_coords.append(event.pos)

			elif event.type == pg.MOUSEBUTTONUP and left_btn_down and user_drawing_mode:
				user_drawing_coords.append(event.pos)
				left_btn_down = False

			elif event.type == pg.KEYDOWN:
				match event.key:
					case pg.K_UP | pg.K_DOWN:
						if event.key == pg.K_UP and fourier:
							n_epicycles = min(n_epicycles * 2, len(fourier))
							print(f'\nNumber of epicycles = {n_epicycles}', end='')
							print(' (max)' if n_epicycles == len(fourier) else '')
						elif event.key == pg.K_DOWN and fourier:
							pow2 = 2 ** int(np.log2(n_epicycles))
							n_epicycles = pow2 // 2 if pow2 == n_epicycles else pow2
							n_epicycles = max(n_epicycles, 2)
							print(f'\nNumber of epicycles = {n_epicycles}')
						path, time = [], 0
						continue
					case pg.K_p:
						# fourier = compute_fourier_from_coords(PI, 'pi symbol')
						fourier = load_fourier_from_file('dft_pi.txt', 'pi symbol')
					case pg.K_g:
						# fourier = compute_fourier_from_coords(GUITAR, 'guitar')
						fourier = load_fourier_from_file('dft_guitar.txt', 'guitar')
					case pg.K_t:
						# fourier = compute_fourier_from_coords(T_REX, 'T. Rex')
						fourier = load_fourier_from_file('dft_t_rex.txt', 'T. Rex')
					case pg.K_c:
						# fourier = compute_fourier_from_coords(COLOSSEUM, 'Colosseum')
						fourier = load_fourier_from_file('dft_colosseum.txt', 'Colosseum')
					case pg.K_SPACE:
						paused = not paused
						continue
					case _:
						continue
				user_drawing_mode = paused = False
				path, time = [], 0
				dt = 2 * np.pi / len(fourier)
				n_epicycles = len(fourier)

		if paused and not user_drawing_mode: continue

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
				path, time = [], 0

		pg.display.update()
		clock.tick(FPS)

if __name__ == '__main__':
	main()
