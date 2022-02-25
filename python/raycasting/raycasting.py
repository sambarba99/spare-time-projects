# Raycasting demo
# Author: Sam Barba
# Created 08/03/2022

# WASD: Move around
# T: Toggle POV/bird's-eye view
# R: Reset

import numpy as np
import pygame as pg
import sys

WIDTH = 1350
HEIGHT = 900
RAY_MAX_LENGTH = (WIDTH ** 2 + HEIGHT ** 2) ** 0.5
FOV_ANGLE = np.deg2rad(90)  # Field of View angle

pov_mode = True  # False = bird's-eye view
rays = []
walls = []

# Start at centre, looking east
player_x = WIDTH / 2
player_y = HEIGHT / 2
player_heading = 0

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def generate_walls(num_walls=5):
	global walls

	walls = []

	for _ in range(num_walls):
		start_x = np.random.randint(0, WIDTH)
		start_y = np.random.randint(0, HEIGHT)
		if np.random.random() < 0.5:
			end_x = start_x
			end_y = start_y + np.random.choice([-200, 200])
		else:
			end_y = start_y
			end_x = start_x + np.random.choice([-200, 200])

		walls.append({"x1": start_x, "y1": start_y, "x2": end_x, "y2": end_y})

	# World border
	walls.append({"x1": 0, "y1": 0, "x2": WIDTH, "y2": 0})
	walls.append({"x1": WIDTH - 2, "y1": 0, "x2": WIDTH - 2, "y2": HEIGHT})
	walls.append({"x1": 0, "y1": HEIGHT - 2, "x2": WIDTH, "y2": HEIGHT - 2})
	walls.append({"x1": 0, "y1": 0, "x2": 0, "y2": HEIGHT})

def generate_rays():
	global player_x, player_y, player_heading, rays, walls

	rays = []

	for a in np.arange(-FOV_ANGLE / 2, FOV_ANGLE / 2, np.deg2rad(0.2)):
		end_x = RAY_MAX_LENGTH * np.cos(a + player_heading) + player_x
		end_y = RAY_MAX_LENGTH * np.sin(a + player_heading) + player_y

		# Assume ray doesn't hit a wall (length = None)
		ray = {"x1": player_x, "y1": player_y, "x2": end_x, "y2": end_y, "length": None}

		for wall in walls:
			intersection = find_intersection(ray, wall)
			if intersection:
				ray["x2"], ray["y2"] = intersection
				ray["length"] = ((ray["x1"] - ray["x2"]) ** 2 + (ray["y1"] - ray["y2"]) ** 2) ** 0.5

		rays.append(ray)

def find_intersection(ray, wall):
	rx1, ry1 = ray["x1"], ray["y1"]
	rx2, ry2 = ray["x2"], ray["y2"]
	wx1, wy1 = wall["x1"], wall["y1"]
	wx2, wy2 = wall["x2"], wall["y2"]

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

def draw_pov_mode():
	global rays

	scene.fill((0, 0, 0))

	wall_segment_width = WIDTH // len(rays)

	ray_lengths = [r["length"] for r in rays if r["length"]]
	longest = max(ray_lengths) if ray_lengths else RAY_MAX_LENGTH

	for idx, r in enumerate(rays):
		d = r["length"]

		h = 0 if not d else map_range(d ** 0.5, 0, RAY_MAX_LENGTH ** 0.5, HEIGHT, HEIGHT * 0.1)
		y = (HEIGHT - h) / 2  # Draw rect from centre
		c = 0 if not d else map_range(d, 0, longest, 255, 20)

		pg.draw.rect(scene, (c, c, c), pg.Rect(idx * wall_segment_width, y, wall_segment_width, h))

	pg.display.update()

def map_range(x, from_lo, from_hi, to_lo, to_hi):
	if from_hi - from_lo == 0:
		return to_hi
	return (x - from_lo) / (from_hi - from_lo) * (to_hi - to_lo) + to_lo

def draw_birds_eye_mode():
	global rays, walls

	scene.fill((0, 0, 0))

	for r in rays:
		pg.draw.line(scene, (220, 220, 220), (r["x1"], r["y1"]), (r["x2"], r["y2"]))

	for w in walls:
		pg.draw.line(scene, (255, 60, 0), (w["x1"], w["y1"]), (w["x2"], w["y2"]), width=4)

	pg.display.update()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

pg.init()
pg.display.set_caption("Raycasting demo")
scene = pg.display.set_mode((WIDTH, HEIGHT))

generate_walls()
generate_rays()
draw_pov_mode()

done = True
key = None

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)
		elif event.type == pg.KEYDOWN:
			done = False
			key = event.key

			if event.key == pg.K_r:  # Reset
				player_x = WIDTH / 2
				player_y = HEIGHT / 2
				player_heading = 0
				generate_walls()
			elif event.key == pg.K_t:  # Toggle view mode
				pov_mode = not pov_mode
		elif event.type == pg.KEYUP:
			done = True

	if not done:
		if key == pg.K_w:  # Move forwards
			dx = 4 * np.cos(player_heading)
			dy = 4 * np.sin(player_heading)
			if 2 <= player_x + dx < WIDTH - 2 and 2 <= player_y + dy < HEIGHT - 2:
				player_x += dx
				player_y += dy

		elif key == pg.K_s:  # Move backwards
			dx = 4 * np.cos(player_heading)
			dy = 4 * np.sin(player_heading)
			if 2 <= player_x - dx < WIDTH - 2 and 2 <= player_y - dy < HEIGHT - 2:
				player_x -= dx
				player_y -= dy

		elif key == pg.K_a:  # Turn left
			player_heading -= np.deg2rad(1)

		elif key == pg.K_d:  # Turn right
			player_heading += np.deg2rad(1)

		generate_rays()

		if pov_mode:
			draw_pov_mode()
		else:
			draw_birds_eye_mode()
