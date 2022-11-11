"""
Ray casting demo

Author: Sam Barba
Created 08/03/2022

Controls:
WASD: move around
R: reset
"""

import sys

import numpy as np
import pygame as pg

# World constants
WIDTH, HEIGHT = 1600, 800
GRID_COLS, GRID_ROWS = 20, 10  # World is a 20x10 grid
GRID_SQUARE_SIZE = WIDTH // GRID_COLS  # HEIGHT // GRID_ROWS
BORDER_LIM = 10

# Ray/rendering constants
N_RAYS = 400
FOV_ANGLE = np.pi / 3  # 60 deg
DELTA_ANGLE = FOV_ANGLE / N_RAYS
SCREEN_DIST = WIDTH * 0.5 / np.tan(FOV_ANGLE * 0.5)
WALL_WIDTH = WIDTH / N_RAYS  # Width of each wall segment to render in 3D
PROJ_HEIGHT_SCALE = 50
MINIMAP_SCALE = 0.2

# Player constants
MOVEMENT_SPEED = 8
TURNING_SPEED = 2.5
FPS = 30

walls = []
ray_casting_result = []

# Start at top-left, looking towards centre
player_x = player_y = BORDER_LIM
player_heading = np.arctan2(HEIGHT, WIDTH)
scene = None

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def generate_walls(n_boxes=15):
	"""Choose random indices of the grid world, and draw boxes there"""

	def make_box(grid_idx):
		x_top_left = (grid_idx % GRID_COLS) * GRID_SQUARE_SIZE
		y_top_left = (grid_idx // GRID_COLS) * GRID_SQUARE_SIZE
		x_bottom_right = x_top_left + GRID_SQUARE_SIZE
		y_bottom_right = y_top_left + GRID_SQUARE_SIZE
		wall1 = {'x1': x_top_left, 'y1': y_top_left, 'x2': x_bottom_right, 'y2': y_top_left}
		wall2 = {'x1': x_bottom_right, 'y1': y_top_left, 'x2': x_bottom_right, 'y2': y_bottom_right}
		wall3 = {'x1': x_bottom_right, 'y1': y_bottom_right, 'x2': x_top_left, 'y2': y_bottom_right}
		wall4 = {'x1': x_top_left, 'y1': y_bottom_right, 'x2': x_top_left, 'y2': y_top_left}
		return [wall1, wall2, wall3, wall4]

	global walls

	walls = []
	grid_indices = np.random.choice(GRID_ROWS * GRID_COLS, size=n_boxes, replace=False)
	for idx in grid_indices:
		walls.extend(make_box(idx))

	# World border
	walls.append({'x1': 0, 'y1': 0, 'x2': WIDTH, 'y2': 0})
	walls.append({'x1': WIDTH, 'y1': 0, 'x2': WIDTH, 'y2': HEIGHT})
	walls.append({'x1': 0, 'y1': HEIGHT, 'x2': WIDTH, 'y2': HEIGHT})
	walls.append({'x1': 0, 'y1': 0, 'x2': 0, 'y2': HEIGHT})

def generate_rays():
	def find_intersection(ray, wall):
		rx1, ry1 = ray['x1'], ray['y1']
		rx2, ry2 = ray['x2'], ray['y2']
		wx1, wy1 = wall['x1'], wall['y1']
		wx2, wy2 = wall['x2'], wall['y2']

		denom = (rx1 - rx2) * (wy1 - wy2) - (ry1 - ry2) * (wx1 - wx2)
		if denom == 0:
			return None

		t = ((rx1 - wx1) * (wy1 - wy2) - (ry1 - wy1) * (wx1 - wx2)) / denom
		u = -((rx1 - rx2) * (ry1 - wy1) - (ry1 - ry2) * (rx1 - wx1)) / denom

		if 0 <= t <= 1 and 0 <= u <= 1:
			intersection_x = rx1 + t * (rx2 - rx1)
			intersection_y = ry1 + t * (ry2 - ry1)
			return intersection_x, intersection_y

		return None

	global ray_casting_result

	ray_casting_result = []
	ray_angle = player_heading - FOV_ANGLE / 2
	for _ in range(N_RAYS):
		end_x = 1e6 * np.cos(ray_angle) + player_x
		end_y = 1e6 * np.sin(ray_angle) + player_y

		# Initialise ray withouth 'length' attribute; this is calculated in loop below
		ray = {'x1': player_x, 'y1': player_y, 'x2': end_x, 'y2': end_y}

		# At the end of this loop, ray['length'] will be the distance to the nearest wall that the ray hits
		for wall in walls:
			intersection = find_intersection(ray, wall)
			if intersection:
				ray['x2'], ray['y2'] = intersection
				ray['length'] = ((ray['x1'] - ray['x2']) ** 2 + (ray['y1'] - ray['y2']) ** 2) ** 0.5

		distorted_dist = ray['length']
		correct_dist = distorted_dist * np.cos(player_heading - ray_angle)  # Remove fish eye distortion
		correct_dist += 1e-6  # Ensure nonzero
		proj_height = SCREEN_DIST / correct_dist * PROJ_HEIGHT_SCALE
		proj_height = min(HEIGHT, proj_height)

		ray_casting_result.append((ray, proj_height))

		ray_angle += DELTA_ANGLE

def draw_pov_mode():
	def map_range(x, from_lo, from_hi, to_lo, to_hi):
		"""Map x from [from_lo, from_hi] to [to_lo, to_hi]"""

		if from_hi - from_lo == 0:
			return to_hi
		return (x - from_lo) / (from_hi - from_lo) * (to_hi - to_lo) + to_lo

	# Draw sky and ground
	pg.draw.rect(scene, (20, 100, 255), pg.Rect(0, 0, WIDTH, HEIGHT / 2))
	pg.draw.rect(scene, (20, 150, 20), pg.Rect(0, HEIGHT / 2, WIDTH, HEIGHT / 2))

	for idx, (_, proj_height) in enumerate(ray_casting_result):
		c = map_range(proj_height ** 0.5, 0, HEIGHT ** 0.5, 20, 255)
		y = (HEIGHT - proj_height) / 2  # Centre wall vertically

		pg.draw.rect(scene, (c, c, c), pg.Rect(idx * WALL_WIDTH, y, WALL_WIDTH, proj_height))

def draw_minimap():
	for idx, (ray, _) in enumerate(ray_casting_result):
		if idx % 10 == 0:
			pg.draw.line(
				scene,
				(255, 0, 0),
				(ray['x1'] * MINIMAP_SCALE, ray['y1'] * MINIMAP_SCALE), (ray['x2'] * MINIMAP_SCALE, ray['y2'] * MINIMAP_SCALE)
			)

	pg.draw.circle(scene, (255, 0, 0), (player_x * MINIMAP_SCALE, player_y * MINIMAP_SCALE), 4)

	for idx, w in enumerate(walls):
		pg.draw.line(
			scene,
			(255, 255, 255),
			(w['x1'] * MINIMAP_SCALE, w['y1'] * MINIMAP_SCALE), (w['x2'] * MINIMAP_SCALE, w['y2'] * MINIMAP_SCALE)
		)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	global player_heading, player_x, player_y, scene

	pg.init()
	pg.display.set_caption('Ray casting demo')
	scene = pg.display.set_mode((WIDTH, HEIGHT))
	clock = pg.time.Clock()

	generate_walls()
	generate_rays()
	draw_pov_mode()
	draw_minimap()

	key_pressed = None

	while True:
		for event in pg.event.get():
			match event.type:
				case pg.QUIT: sys.exit()
				case pg.KEYDOWN:
					if event.key == pg.K_r:  # Reset
						player_x = player_y = BORDER_LIM
						player_heading = np.arctan2(HEIGHT, WIDTH)
						generate_walls()
					else:
						key_pressed = event.key
				case pg.KEYUP:
					key_pressed = None

		match key_pressed:
			case pg.K_w:  # Move forwards
				dx = MOVEMENT_SPEED * np.cos(player_heading)
				dy = MOVEMENT_SPEED * np.sin(player_heading)
				if BORDER_LIM <= player_x + dx < WIDTH - BORDER_LIM and BORDER_LIM <= player_y + dy < HEIGHT - BORDER_LIM:
					player_x += dx
					player_y += dy
			case pg.K_s:  # Move backwards
				dx = MOVEMENT_SPEED * np.cos(player_heading)
				dy = MOVEMENT_SPEED * np.sin(player_heading)
				if BORDER_LIM <= player_x - dx < WIDTH - BORDER_LIM and BORDER_LIM <= player_y - dy < HEIGHT - BORDER_LIM:
					player_x -= dx
					player_y -= dy
			case pg.K_a:  # Turn left
				player_heading -= np.deg2rad(TURNING_SPEED)
			case pg.K_d:  # Turn right
				player_heading += np.deg2rad(TURNING_SPEED)

		generate_rays()
		draw_pov_mode()
		draw_minimap()
		pg.display.update()
		clock.tick(FPS)

if __name__ == '__main__':
	main()
