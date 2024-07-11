"""
Game environment functionality

Author: Sam Barba
Created 11/07/2024
"""

from math import pi, sin, cos, atan2

import numpy as np
import pygame as pg

from pytorch_ppo_asteroids.game_env.constants import *


vec2 = pg.math.Vector2


def lines_intersect(line_a, line_b):
	(ax1, ay1), (ax2, ay2) = line_a
	(bx1, by1), (bx2, by2) = line_b

	denom = (ax1 - ax2) * (by1 - by2) - (ay1 - ay2) * (bx1 - bx2)

	if denom:
		t = ((ax1 - bx1) * (by1 - by2) - (ay1 - by1) * (bx1 - bx2)) / denom
		u = -((ax1 - ax2) * (ay1 - by1) - (ay1 - ay2) * (ax1 - bx1)) / denom

		return 0 <= t <= 1 and 0 <= u <= 1

	return False


class Spaceship:
	def __init__(self):
		# Physical properties
		self.pos = vec2(SCENE_WIDTH / 2, SCENE_HEIGHT / 2)
		self.vel = vec2()  # 0,0
		self.acc = 0
		self.heading = -pi / 2  # Point up (heading = direction spaceship is pointing)
		self.lasers = []
		self.score = 0

		# Lines defining the spaceship (hull lines + thruster lines)
		self.lines = None
		self.update_lines()

	def perform_action(self, action):
		"""Perform an action e.g. boosting, and update spaceship properties accordingly"""

		# 1. Decode action num

		boosting = action in (1, 5, 6, 7, 10, 11)
		turning_left = action in (2, 5, 8, 10)
		turning_right = action in (3, 6, 9, 11)
		shooting = action in (4, 7, 8, 9, 10, 11)

		# 2. Apply force to accelerate/decelerate if necessary

		self.acc = ACCELERATION_FORCE if boosting else 0

		# 3. Apply steering if necessary

		if turning_left:
			self.heading = (self.heading - TURN_RATE) % (2 * pi)
		elif turning_right:
			self.heading = (self.heading + TURN_RATE) % (2 * pi)

		# 4. Update velocity

		if self.acc:
			acc_vector = vec2(cos(self.heading), sin(self.heading)) * self.acc
			self.vel += acc_vector
			self.vel.clamp_magnitude_ip(MAX_VEL)

		# 5. Update position

		self.pos += self.vel
		self.pos.x %= SCENE_WIDTH
		self.pos.y %= SCENE_HEIGHT
		self.update_lines()

		# 6. Add lasers

		if shooting and len(self.lasers) < MAX_LASERS:
			self.lasers.append(
				Laser(
					vec2(self.lines[0][0]),  # Shoot from front of spaceship
					self.heading
				)
			)

	def check_collision(self, asteroid):
		for s1, s2 in self.lines[:3]:  # Only check hull lines
			for a1, a2 in asteroid.lines:
				if lines_intersect((s1, s2), (a1, a2)):
					return True
		return False

	def update_lines(self):
		l1 = (
			(self.pos.x + SPACESHIP_SCALE * cos(self.heading),
			self.pos.y + SPACESHIP_SCALE * sin(self.heading)),
			(self.pos.x - SPACESHIP_SCALE * cos(0.66 + self.heading) * 0.95,
			self.pos.y - SPACESHIP_SCALE * sin(0.66 + self.heading) * 0.95)
		)
		l2 = (
			l1[0],
			(self.pos.x - SPACESHIP_SCALE * cos(0.66 - self.heading) * 0.95,
			self.pos.y + SPACESHIP_SCALE * sin(0.66 - self.heading) * 0.95)
		)
		l3 = (
			(self.pos.x - SPACESHIP_SCALE * cos(self.heading + pi / 4) * 0.71,
			self.pos.y - SPACESHIP_SCALE * sin(self.heading + pi / 4) * 0.71),
			(self.pos.x - SPACESHIP_SCALE * cos(-self.heading + pi / 4) * 0.71,
			self.pos.y + SPACESHIP_SCALE * sin(-self.heading + pi / 4) * 0.71)
		)
		self.lines = [l1, l2, l3]  # Hull lines
		if self.acc:
			# Thruster lines
			t1 = (
				(self.pos.x - SPACESHIP_SCALE * cos(self.heading),
				self.pos.y - SPACESHIP_SCALE * sin(self.heading)),
				(self.pos.x - SPACESHIP_SCALE * cos(self.heading + pi / 6) * 0.63,
				self.pos.y - SPACESHIP_SCALE * sin(self.heading + pi / 6) * 0.63)
			)
			t2 = (
				t1[0],
				(self.pos.x - SPACESHIP_SCALE * cos(-self.heading + pi / 6) * 0.63,
				self.pos.y + SPACESHIP_SCALE * sin(-self.heading + pi / 6) * 0.63)
			)
			self.lines.extend([t1, t2])


class Asteroid:
	def __init__(self, random_obj, size, pos, direction=None):
		self.pos = pos
		self.size = size
		self.radius = ASTEROID_RADII[size]
		self.direction = random_obj.uniform(0, 2 * pi) if direction is None else direction
		dir_vector = vec2(cos(self.direction), sin(self.direction))
		vel_magnitude = random_obj.uniform(ASTEROID_VELS[size] * 0.1, ASTEROID_VELS[size])
		self.vel = dir_vector * vel_magnitude

		# Generate random points around the centre, then use these to generate lines
		self.num_points = random_obj.randint(5, 12)
		self.rand_offsets = [random_obj.uniform(-self.radius * 0.3, self.radius * 0.3) for _ in range(self.num_points)]
		self.lines = None
		self.update()

	def update(self):
		self.pos += self.vel
		self.pos.x %= SCENE_WIDTH
		self.pos.y %= SCENE_HEIGHT

		points = []
		for i in range(self.num_points):
			angle = i / self.num_points * 2 * pi
			r = self.radius + self.rand_offsets[i]
			points.append((self.pos.x + r * cos(angle), self.pos.y + r * sin(angle)))

		self.lines = [(p1, p2) for p1, p2 in zip(points[:-1], points[1:])]
		self.lines.append((points[-1], points[0]))


class Laser:
	def __init__(self, pos, direction):
		self.pos = pos
		self.last_pos = pos
		dir_vector = vec2(cos(direction), sin(direction))
		self.vel = dir_vector * LASER_VEL
		self.life = LASER_LIFESPAN

	def update(self):
		self.life -= 1
		if self.life == 0:
			return
		self.last_pos = self.pos.copy()
		self.pos += self.vel
		if self.pos.x < 0:
			self.pos.x = self.last_pos.x = SCENE_WIDTH
		elif self.pos.x > SCENE_WIDTH:
			self.pos.x = self.last_pos.x = 0
		if self.pos.y < 0:
			self.pos.y = self.last_pos.y = SCENE_HEIGHT
		elif self.pos.y > SCENE_HEIGHT:
			self.pos.y = self.last_pos.y = 0

	def check_asteroid_hit(self, asteroid):
		for a1, a2 in asteroid.lines:
			if lines_intersect((self.pos, self.last_pos), (a1, a2)):
				return True
		return False


class GameEnv:
	def __init__(self, *, random_obj, do_rendering, num_init_asteroids, ts_reward, training_mode=False):
		self.rand = random_obj
		self.training_mode = training_mode
		self.spaceship = None
		self.asteroids = None
		self.level = None
		self.state_update_needed = None
		self.num_init_asteroids = num_init_asteroids
		self.ts_reward = ts_reward

		# Stores the indices of the nearest MAX_ASTEROIDS_DETECT asteroids
		self.detected_asteroid_idx = None

		if do_rendering:
			pg.init()
			pg.display.set_caption('PPO Asteroids player')
			self.font = pg.font.SysFont('consolas', 20)
			self.scene = pg.display.set_mode((SCENE_WIDTH, SCENE_HEIGHT))
			self.clock = pg.time.Clock()

		self.reset()

	def get_state(self):
		"""Obtain the state of the spaceship/environment"""

		def toroidal_distance_to_asteroid(asteroid):
			"""Toroidal distance from the spaceship to a given asteroid"""

			dx_abs = abs(self.spaceship.pos.x - asteroid.pos.x)
			dy_abs = abs(self.spaceship.pos.y - asteroid.pos.y)
			dx = min(dx_abs, SCENE_WIDTH - dx_abs)
			dy = min(dy_abs, SCENE_HEIGHT - dy_abs)

			return dx * dx + dy * dy

		def toroidal_angles_to_asteroid(asteroid, spaceship_heading, spaceship_direction):
			"""Toroidal heading/direction from the spaceship to a given asteroid"""

			dx = asteroid.pos.x - self.spaceship.pos.x
			dy = asteroid.pos.y - self.spaceship.pos.y

			# Wrap-around distances
			dx_wrap_x = SCENE_WIDTH - abs(dx) if dx != 0 else 0
			dy_wrap_y = SCENE_HEIGHT - abs(dy) if dy != 0 else 0

			# Possible positions considering wrap-around
			possible_positions = [
				(dx, dy),  # No wrap
				(-dx_wrap_x if dx > 0 else dx_wrap_x, dy),  # Wrap x-axis
				(dx, -dy_wrap_y if dy > 0 else dy_wrap_y),  # Wrap y-axis
				(-dx_wrap_x if dx > 0 else dx_wrap_x, -dy_wrap_y if dy > 0 else dy_wrap_y)  # Wrap both axes
			]

			# Find the closest position considering wrap-around
			closest_pos = min(possible_positions, key=lambda pos: pos[0] * pos[0] + pos[1] * pos[1])

			# Calculate the angle to the closest position
			angle_to_asteroid = atan2(closest_pos[1], closest_pos[0])

			# -pi to pi
			heading_to_asteroid = (angle_to_asteroid - spaceship_heading + pi / 2) % (2 * pi)
			if heading_to_asteroid > pi:
				heading_to_asteroid -= 2 * pi
			direction_to_asteroid = (angle_to_asteroid - spaceship_direction + pi / 2) % (2 * pi)
			if direction_to_asteroid > pi:
				direction_to_asteroid -= 2 * pi

			return heading_to_asteroid, direction_to_asteroid

		def is_collision_course(asteroid, lookahead_timesteps=100):
			"""Check if the spaceship will collide with a given asteroid in the next N timesteps"""

			a_pos = asteroid.pos.copy()
			s_pos = self.spaceship.pos.copy()
			d_collision = asteroid.radius + (min(asteroid.rand_offsets) + max(asteroid.rand_offsets)) / 2 + SPACESHIP_SCALE * 1.8

			for _ in range(lookahead_timesteps):
				if a_pos.distance_to(s_pos) < d_collision:
					return True
				a_pos += asteroid.vel
				a_pos.x %= SCENE_WIDTH
				a_pos.y %= SCENE_HEIGHT
				s_pos += self.spaceship.vel
				s_pos.x %= SCENE_WIDTH
				s_pos.y %= SCENE_HEIGHT

			return False


		# 1. Caluclate distances to all asteroids, and keep top MAX_ASTEROIDS_DETECT nearest (pad with 0s if necessary)

		if self.state_update_needed:
			distances_to_asteroids = np.array([toroidal_distance_to_asteroid(a) for a in self.asteroids])
			self.detected_asteroid_idx = np.argsort(distances_to_asteroids)[:MAX_ASTEROIDS_DETECT]
			distances_to_asteroids = distances_to_asteroids[self.detected_asteroid_idx]
			self.state_update_needed = False
		else:
			distances_to_asteroids = np.array([toroidal_distance_to_asteroid(self.asteroids[idx]) for idx in self.detected_asteroid_idx])
		pad_size = max(0, MAX_ASTEROIDS_DETECT - len(self.detected_asteroid_idx))

		# 2. Calculate angles to these asteroids

		# Convert spaceship heading and direction to [-pi, pi] range
		adjusted_heading = (self.spaceship.heading + pi / 2) % (2 * pi)
		if adjusted_heading > pi:
			adjusted_heading -= 2 * pi
		adjusted_direction = (atan2(self.spaceship.vel.y, self.spaceship.vel.x) + pi / 2) % (2 * pi)
		if adjusted_direction > pi:
			adjusted_direction -= 2 * pi

		angles_to_asteroids = [
			toroidal_angles_to_asteroid(self.asteroids[idx], adjusted_heading, adjusted_direction)
			for idx in self.detected_asteroid_idx
		]
		headings_to_asteroids, directions_to_asteroids = zip(*angles_to_asteroids)

		# 3. Calculate relative velocities of these asteroids

		rel_asteroid_vel_mags, rel_asteroid_vel_angles = [], []
		for idx in self.detected_asteroid_idx:
			asteroid = self.asteroids[idx]
			mag = asteroid.vel.magnitude() - self.spaceship.vel.magnitude()
			angle = (atan2(asteroid.vel.y, asteroid.vel.x) - atan2(self.spaceship.vel.y, self.spaceship.vel.x)) % (2 * pi)
			if angle > pi:
				angle -= 2 * pi
			rel_asteroid_vel_mags.append(mag)
			rel_asteroid_vel_angles.append(angle)

		# 4. Agent knows these (normalised):

		distances_to_asteroids = np.pad(distances_to_asteroids, (0, pad_size)) / MAX_ASTEROID_DIST
		headings_to_asteroids = np.pad(headings_to_asteroids, (0, pad_size)) / pi
		directions_to_asteroids = np.pad(directions_to_asteroids, (0, pad_size)) / pi
		rel_asteroid_vel_mags = np.pad(rel_asteroid_vel_mags, (0, pad_size)) / MAX_VEL
		rel_asteroid_vel_angles = np.pad(rel_asteroid_vel_angles, (0, pad_size)) / pi
		is_collisions = [is_collision_course(self.asteroids[idx]) for idx in self.detected_asteroid_idx]
		is_collisions = np.pad(np.array(is_collisions, dtype=int), (0, pad_size))

		state = [
			*distances_to_asteroids,
			*headings_to_asteroids,
			*directions_to_asteroids,
			*rel_asteroid_vel_mags,
			*rel_asteroid_vel_angles,
			*is_collisions
		]

		return state

	def step(self, action):
		"""
		Perform an action, then return resulting info as the tuple (return, next_state, terminal)
		"""

		self.spaceship.perform_action(action)

		# Update asteroids and lasers
		for asteroid in self.asteroids:
			asteroid.update()
		for laser in self.spaceship.lasers:
			laser.update()
			if laser.life == 0:
				self.spaceship.lasers.remove(laser)

		next_state = self.get_state()

		# Check for spaceship-asteroid collision
		if any(self.spaceship.check_collision(asteroid) for asteroid in self.asteroids):
			return COLLISION_PENALTY, next_state, True

		# Check if a laser has hit an asteroid
		for laser in self.spaceship.lasers:
			for idx, asteroid in enumerate(self.asteroids):
				if laser.check_asteroid_hit(asteroid):
					self.asteroids.remove(asteroid)
					self.spaceship.lasers.remove(laser)
					self.spaceship.score += HUMAN_ASTEROID_DESTROY_REWARDS[asteroid.size]
					if asteroid.size != 'small':
						# Split asteroid into 2
						for _ in range(2):
							self.asteroids.append(
								Asteroid(
									self.rand,
									'medium' if asteroid.size == 'large' else 'small',
									asteroid.pos.copy(),
									(asteroid.direction + self.rand.uniform(-pi / 8, pi / 8)) % (2 * pi)
								)
							)

					if len(self.asteroids) == 0:  # Destroyed all asteroids, go to next level
						self.level += 1
						self.spaceship.lasers = []
						# self.add_asteroids(self.level + (7 if self.training_mode else 3))
						self.add_asteroids(self.level + self.num_init_asteroids - 1)

					self.state_update_needed = True

					# Don't reward accidental asteroid hits
					if idx in self.detected_asteroid_idx:
						return AGENT_ASTEROID_DESTROY_REWARD, next_state, False
					else:
						return self.ts_reward, next_state, False

		return self.ts_reward, next_state, False

	def render(self, action, render_meta):
		self.scene.fill('black')

		# Draw spaceship
		flame_green = self.rand.randrange(256)
		for idx, (s1, s2) in enumerate(self.spaceship.lines):
			colour = (255, 255, 255) if idx < 3 else (255, flame_green, 0)  # Thruster (last 2 lines) = red/orange/yellow
			pg.draw.line(self.scene, colour, s1, s2, 2)

		# Draw asteroids
		for idx, asteroid in enumerate(self.asteroids):
			if render_meta and idx not in self.detected_asteroid_idx:
				continue
			for a1, a2 in asteroid.lines:
				colour = (255, 0, 0) if render_meta else (255, 255, 255)
				thickness = 3 if render_meta else 1
				pg.draw.line(self.scene, colour, a1, a2, thickness)

		# Draw lasers
		for laser in self.spaceship.lasers:
			pg.draw.circle(self.scene, (255, 255, 255), laser.pos, 3)

		# Display arrow/space key control (key is green if used, else grey)
		up_colour = (0, 255, 0) if action in (1, 5, 6, 7, 10, 11) else (50, 50, 50)
		left_colour = (0, 255, 0) if action in (2, 5, 8, 10) else (50, 50, 50)
		right_colour = (0, 255, 0) if action in (3, 6, 9, 11) else (50, 50, 50)
		space_colour = (0, 255, 0) if action in (4, 7, 8, 9, 10, 11) else (50, 50, 50)
		pg.draw.rect(self.scene, up_colour, pg.Rect(120, 60, 25, 25))
		pg.draw.rect(self.scene, left_colour, pg.Rect(95, 85, 25, 25))
		pg.draw.rect(self.scene, right_colour, pg.Rect(145, 85, 25, 25))
		pg.draw.rect(self.scene, space_colour, pg.Rect(12, 85, 75, 25))

		# Display level and score
		level_lbl = self.font.render(f'Level: {self.level}', True, (255, 255, 255))
		score_lbl = self.font.render(f'Score: {self.spaceship.score}', True, (255, 255, 255))
		self.scene.blit(level_lbl, (10, 10))
		self.scene.blit(score_lbl, (10, 35))

		pg.display.update()
		self.clock.tick(FPS)

	def add_asteroids(self, n):
		# Don't spawn asteroids around where the spaceship is (also considering wrap-around)
		possible_x = [x for x in range(SCENE_WIDTH) if min(abs(x - self.spaceship.pos.x), SCENE_WIDTH - abs(x - self.spaceship.pos.x)) > 250]
		possible_y = [y for y in range(SCENE_HEIGHT) if min(abs(y - self.spaceship.pos.y), SCENE_HEIGHT - abs(y - self.spaceship.pos.y)) > 250]

		for i in range(n):
			match i % 4:
				case 0: asteroid = Asteroid(self.rand, 'large', vec2(y=self.rand.choice(possible_y)))
				case 1: asteroid = Asteroid(self.rand, 'large', vec2(SCENE_WIDTH - 1, self.rand.choice(possible_y)))
				case 2: asteroid = Asteroid(self.rand, 'large', vec2(self.rand.choice(possible_x), 0))
				case _: asteroid = Asteroid(self.rand, 'large', vec2(self.rand.choice(possible_x), SCENE_HEIGHT - 1))

			self.asteroids.append(asteroid)

	def reset(self):
		# If training, initialise spaceship with a random position/orientation/velocity each time (more variation in initial states)
		self.spaceship = Spaceship()
		if self.training_mode:
			self.spaceship.pos = vec2(self.rand.randrange(SCENE_WIDTH), self.rand.randrange(SCENE_HEIGHT))
			self.spaceship.heading = self.rand.uniform(0, 2 * pi)
			rand_direction = self.rand.uniform(0, 2 * pi)
			rand_magnitude = self.rand.uniform(0, MAX_VEL)
			self.spaceship.vel = vec2(cos(rand_direction), sin(rand_direction)) * rand_magnitude
		self.level = 1
		self.asteroids = []
		self.add_asteroids(self.num_init_asteroids)
		# self.add_asteroids(8 if self.training_mode else 4)
		self.state_update_needed = True
