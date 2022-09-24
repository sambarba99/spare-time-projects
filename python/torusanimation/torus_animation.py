"""
Revolving torus animation (press space to toggle animation)

Author: Sam Barba
Created 09/06/2022
"""

# Faster than np.sin, np.cos
from math import sin, cos
import numpy as np
import pygame as pg
import sys

SIZE = 70  # Rows = cols
PIXEL_SIZE = 12

R1 = 2  # Radius of torus cross-sectional circle
R2 = 5  # Distance from origin to centre of cross-sectional circle

"""
Calculate K1 based on screen size: the maximum x-distance occurs roughly at the edge of the torus,
which is at x = R1 + R2, z = 0. We want that to be displaced 3/8ths of the width of the screen.
"""

K2 = 100  # Arbitrary distance from torus to viewer
K1 = SIZE * K2 * 3 / (8 * (R1 + R2))

CHARS = '.,-~:;!*=#$@'  # 'Dimmest' to 'brighest' chars

# Revolution amounts about x and z axes (increased after each frame, to revolve the torus)
xrev = zrev = 0
scene = font = None
paused = False

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def render_torus(output_grid):
	scene.fill((0, 0, 0))

	for y in range(SIZE):
		for x in range(SIZE):
			lbl = font.render(output_grid[y][x], True, (255, 255, 255))
			lbl_rect = lbl.get_rect(center=(x * PIXEL_SIZE, y * PIXEL_SIZE))
			scene.blit(lbl, lbl_rect)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	global scene, font, xrev, zrev, paused

	pg.init()
	scene = pg.display.set_mode((SIZE * PIXEL_SIZE, SIZE * PIXEL_SIZE))
	pg.display.set_caption('Revolving Torus')
	font = pg.font.SysFont('consolas', 14)

	while True:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				sys.exit()
			elif event.type == pg.KEYDOWN:
				if event.key == pg.K_SPACE:
					paused = not paused

		if paused: continue

		sin_xrev = sin(xrev)
		cos_xrev = cos(xrev)
		sin_zrev = sin(zrev)
		coz_zrev = cos(zrev)

		output_grid = [[' '] * SIZE for _ in range(SIZE)]
		zbuffer = [[0] * SIZE for _ in range(SIZE)]

		# Theta goes around the cross-sectional circle of the torus
		for theta in np.linspace(0, 2 * np.pi, 70):
			sin_theta = sin(theta)
			cos_theta = cos(theta)

			# Phi revolves this circle around the y-axis, creating the torus ('solid of revolution')
			for phi in np.linspace(0, 2 * np.pi, 170):
				sin_phi = sin(phi)
				cos_phi = cos(phi)

				# x, y coordinates before revolving
				circlex = R2 + R1 * cos_theta
				circley = R1 * sin_theta

				# 3D coords after revolution
				x = circlex * (coz_zrev * cos_phi + sin_xrev * sin_zrev * sin_phi) - circley * cos_xrev * sin_zrev
				y = circlex * (sin_zrev * cos_phi - sin_xrev * coz_zrev * sin_phi) + circley * cos_xrev * coz_zrev
				z = K2 + cos_xrev * circlex * sin_phi + circley * sin_xrev
				z_recip = 1 / z

				# x, y projection (y is negated, as y goes up in 3D space but down on 2D displays)
				x_proj = int(SIZE / 2 + K1 * z_recip * x)
				y_proj = -int(SIZE / 2 + K1 * z_recip * y)

				# Luminance (ranges from -root(2) to root(2))
				lum = cos_phi * cos_theta * sin_zrev - cos_xrev * cos_theta * sin_phi - sin_xrev * sin_theta \
					+ coz_zrev * (cos_xrev * sin_theta - cos_theta * sin_xrev * sin_phi)

				# Larger 1/z means the pixel is closer to the viewer than what's already rendered
				if z_recip > zbuffer[y_proj][x_proj]:
					zbuffer[y_proj][x_proj] = z_recip
					# Multiply by 8 to get idx in range 0 - 11 (8 * sqrt(2) = 11.31)
					lum_idx = int(lum * 8)
					output_grid[y_proj][x_proj] = CHARS[max(lum_idx, 0)]

		render_torus(output_grid)
		pg.display.update()
		xrev += 0.04
		zrev += 0.01

if __name__ == '__main__':
	main()
