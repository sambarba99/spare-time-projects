"""
Bézier curve drawing

Author: Sam Barba
Created 09/09/2022

Controls:
Right-click: add a point
Left-click and drag: move a point
R: reset
"""

import sys

import numpy as np
import pygame as pg

SIZE = 800
MAX_POINTS = 5
POINT_RADIUS = 6

scene = None
points = []

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def draw_points_and_curve():
	def draw_connective_lines():
		if len(points) < 2: return

		for point, next_point in zip(points[:-1], points[1:]):
			pg.draw.line(scene, (80, 80, 80), point, next_point)

	scene.fill((0, 0, 0))

	# Draw connective lines, then curve on top, then points on top
	draw_connective_lines()
	draw_curve()
	for point in points:
		pg.draw.circle(scene, (255, 0, 0), point, POINT_RADIUS)

	pg.display.update()

def draw_curve():
	def bezier_point(control_points, t):
		def lerp(a, b, t):
			"""Linear interpolation between (ax,ay) and (bx,by)"""
			ax, ay = a
			bx, by = b
			lx = ax + t * (bx - ax)
			ly = ay + t * (by - ay)
			return lx, ly

		while len(control_points) > 1:
			control_points = zip(control_points[:-1], control_points[1:])
			control_points = [lerp(point, next_point, t) for point, next_point in control_points]

		return control_points[0]

	if len(points) < 2: return

	for t in np.linspace(0, 1, 1500):
		point = bezier_point(points, t)
		point = list(map(int, point))
		scene.set_at(point, (255, 255, 255))

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	global scene, points

	pg.init()
	pg.display.set_caption('Bézier curve drawing')
	scene = pg.display.set_mode((SIZE, SIZE))

	clicked_point_idx = None

	while True:
		for event in pg.event.get():
			match event.type:
				case pg.QUIT: sys.exit()
				case pg.MOUSEBUTTONDOWN:
					x, y = event.pos

					if event.button == 1 and points:  # Left-click to drag point
						point_distances_from_mouse = np.linalg.norm(
							np.array(points) - np.array([x, y]),
							axis=1
						)
						min_dist = min(point_distances_from_mouse)
						if min_dist <= POINT_RADIUS:  # Mouse is on a point
							clicked_point_idx = np.argmin(point_distances_from_mouse)

					elif event.button == 3:  # Right-click to add point
						if len(points) == MAX_POINTS:
							print('Max. no. points added')
							continue

						# Constrain x and y
						x = np.clip(x, POINT_RADIUS, SIZE - POINT_RADIUS)
						y = np.clip(y, POINT_RADIUS, SIZE - POINT_RADIUS)
						points.append([x, y])
						draw_points_and_curve()
				case pg.MOUSEBUTTONUP:
					if event.button == 1:
						clicked_point_idx = None
				case pg.KEYDOWN:
					if event.key == pg.K_r:  # Reset
						points = []
						scene.fill((0, 0, 0))
						pg.display.update()

		if clicked_point_idx is not None:
			points[clicked_point_idx] = list(pg.mouse.get_pos())
			draw_points_and_curve()

if __name__ == '__main__':
	main()
