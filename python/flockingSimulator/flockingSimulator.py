"""
Flocking simulator using boids

Author: Sam Barba
Created 11/03/2022

Controls:
1/2/3: toggle separation/alignment/cohesion
Q/W: +/- perception radius
A/S: +/- max steering force
R: reset parameter values
"""

from boid import Boid
import numpy as np
import pygame as pg
import sys

# Simulation constants
WIDTH, HEIGHT = 1500, 900
FPS = 120
NORTH_ARROW = np.array([[0, 33], [9, 0], [18, 33]]).astype(float)  # Coords describing an arrow

# Boid constant extremes
MIN_PERCEP_RADIUS = 0
MAX_PERCEP_RADIUS = 200
MIN_STEERING_FORCE = 0
MAX_STEERING_FORCE = 4

# Param control sliders (simple vertical lines) - initialise each slider at centre of its scale

SLIDER_MIN_X = 211
SLIDER_MAX_X = 350

percep_radius_slider = [[(SLIDER_MIN_X + SLIDER_MAX_X) / 2, 39 - 7],
	[(SLIDER_MIN_X + SLIDER_MAX_X) / 2, 39 + 7]]
max_steering_force_slider = [[(SLIDER_MIN_X + SLIDER_MAX_X) / 2, 61 - 7],
	[(SLIDER_MIN_X + SLIDER_MAX_X) / 2, 61 + 7]]
max_vel_slider = [[(SLIDER_MIN_X + SLIDER_MAX_X) / 2, 83 - 7],
	[(SLIDER_MIN_X + SLIDER_MAX_X) / 2, 83 + 7]]

do_separation = do_alignment = do_cohesion = True

flock = scene = None

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def generate_boids(n=50):
	global flock

	percep_radius = get_slider_val(percep_radius_slider, MIN_PERCEP_RADIUS, MAX_PERCEP_RADIUS)
	max_steering_force = get_slider_val(max_steering_force_slider, MIN_STEERING_FORCE, MAX_STEERING_FORCE)

	flock = [Boid(percep_radius, max_steering_force, WIDTH, HEIGHT) for _ in range(n)]

def draw():
	global flock, scene

	scene.fill((20, 20, 20))

	for boid in flock:
		arrow = get_oriented_arrow(boid)
		pg.draw.polygon(scene, (220, 220, 220), arrow)

	# Draw controls
	pg.draw.rect(scene, (0, 0, 60), pg.Rect(0, 0, 400, 145))
	font = pg.font.SysFont('consolas', 16)
	controls_lbl = font.render('Controls (1-3, Q/W, A/S, R)', True, (220, 220, 220))
	percep_radius_ctrl_lbl = font.render('Perecption radius:', True, (220, 220, 220))
	max_steering_force_ctrl_lbl = font.render('Max steering force:', True, (220, 220, 220))
	min_percep_radius_lbl = font.render(str(MIN_PERCEP_RADIUS), True, (220, 220, 220))
	max_percep_radius_lbl = font.render(str(MAX_PERCEP_RADIUS), True, (220, 220, 220))
	min_steering_force_lbl = font.render(str(MIN_STEERING_FORCE), True, (220, 220, 220))
	max_steering_force_lbl = font.render(str(MAX_STEERING_FORCE), True, (220, 220, 220))
	do_separation_lbl = font.render(f"Doing separation: {'Yes' if do_separation else 'No'}", True, (220, 220, 220))
	do_alignment_lbl = font.render(f"Doing alignment: {'Yes' if do_alignment else 'No'}", True, (220, 220, 220))
	do_cohesion_lbl = font.render(f"Doing cohesion: {'Yes' if do_cohesion else 'No'}", True, (220, 220, 220))
	scene.blit(controls_lbl, (80, 10))
	scene.blit(percep_radius_ctrl_lbl, (23, 32))
	scene.blit(max_steering_force_ctrl_lbl, (14, 54))
	scene.blit(min_percep_radius_lbl, (195, 32))
	scene.blit(max_percep_radius_lbl, (360, 32))
	scene.blit(min_steering_force_lbl, (195, 54))
	scene.blit(max_steering_force_lbl, (360, 54))
	scene.blit(do_separation_lbl, (32, 76))
	scene.blit(do_alignment_lbl, (41, 98))
	scene.blit(do_cohesion_lbl, (50, 120))

	# Lines representing slider scales
	pg.draw.line(scene, (255, 0, 0), (SLIDER_MIN_X, 39), (SLIDER_MAX_X, 39))
	pg.draw.line(scene, (255, 0, 0), (SLIDER_MIN_X, 61), (SLIDER_MAX_X, 61))

	# Lines representing sliders on their respective scales
	pg.draw.line(scene, (255, 0, 0), *percep_radius_slider)
	pg.draw.line(scene, (255, 0, 0), *max_steering_force_slider)

	pg.display.update()

def get_oriented_arrow(boid):
	arrow_centre = np.mean(NORTH_ARROW, axis=0)

	# Translate arrow so that its centre is at boid's position
	translated_arrow = NORTH_ARROW + boid.pos - arrow_centre

	arrow_centre = np.mean(translated_arrow, axis=0)

	# Heading based on x and y velocity (+ pi / 2 so that heading of 0 means north)
	heading = np.arctan2(boid.vel[1], boid.vel[0]) + np.pi / 2

	# Now rotate arrow by 'heading' (clockwise, hence the minus) about arrow_centre
	theta = -heading
	rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	oriented_arrow = (translated_arrow - arrow_centre).dot(rotate_matrix) + arrow_centre

	return oriented_arrow

def get_slider_val(slider, min_val, max_val):
	slider_x = slider[0][0]
	return (slider_x - SLIDER_MIN_X) / (SLIDER_MAX_X - SLIDER_MIN_X) * (max_val - min_val) + min_val

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	global percep_radius_slider, max_steering_force_slider, max_vel_slider, scene, \
		do_separation, do_alignment, do_cohesion

	pg.init()
	pg.display.set_caption('Flocking Simulator')
	scene = pg.display.set_mode((WIDTH, HEIGHT))
	clock = pg.time.Clock()

	generate_boids()

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

				if key == pg.K_r:  # Reset params (move sliders to centre)
					do_separation = do_alignment = do_cohesion = True

					percep_radius_slider = [[(SLIDER_MIN_X + SLIDER_MAX_X) / 2, 39 - 7],
						[(SLIDER_MIN_X + SLIDER_MAX_X) / 2, 39 + 7]]
					max_steering_force_slider = [[(SLIDER_MIN_X + SLIDER_MAX_X) / 2, 61 - 7],
						[(SLIDER_MIN_X + SLIDER_MAX_X) / 2, 61 + 7]]
					max_vel_slider = [[(SLIDER_MIN_X + SLIDER_MAX_X) / 2, 83 - 7],
						[(SLIDER_MIN_X + SLIDER_MAX_X) / 2, 83 + 7]]
				elif key == pg.K_1:
					do_separation = not do_separation
				elif key == pg.K_2:
					do_alignment = not do_alignment
				elif key == pg.K_3:
					do_cohesion = not do_cohesion

			elif event.type == pg.KEYUP:
				done = True

		if not done:
			if key == pg.K_q:
				if percep_radius_slider[0][0] > SLIDER_MIN_X:
					percep_radius_slider[0][0] -= 1
					percep_radius_slider[1][0] -= 1
			if key == pg.K_w:
				if percep_radius_slider[0][0] < SLIDER_MAX_X:
					percep_radius_slider[0][0] += 1
					percep_radius_slider[1][0] += 1
			if key == pg.K_a:
				if max_steering_force_slider[0][0] > SLIDER_MIN_X:
					max_steering_force_slider[0][0] -= 1
					max_steering_force_slider[1][0] -= 1
			if key == pg.K_s:
				if max_steering_force_slider[0][0] < SLIDER_MAX_X:
					max_steering_force_slider[0][0] += 1
					max_steering_force_slider[1][0] += 1

			perception_radius = get_slider_val(percep_radius_slider, MIN_PERCEP_RADIUS, MAX_PERCEP_RADIUS)
			max_steering_force = get_slider_val(max_steering_force_slider, MIN_STEERING_FORCE, MAX_STEERING_FORCE)

			for boid in flock:
				boid.perception_radius = perception_radius
				boid.max_steering_force = max_steering_force

		for boid in flock:
			boid.apply_behaviour(flock, do_separation, do_alignment, do_cohesion)
		for boid in flock:
			boid.update()

		draw()

		clock.tick(FPS)

if __name__ == '__main__':
	main()
