"""
Game environment functionality

Author: Sam Barba
Created 11/07/2024
"""

from math import pi, sin, cos, atan2

import numpy as np
import pygame as pg
from scipy.optimize import fsolve

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
		self.bullets = []
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

		# 6. Add bullets

		if shooting and len(self.bullets) < MAX_BULLETS:
			self.bullets.append(
				Bullet(
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
		self.size = size
		self.pos = pos
		self.radius = ASTEROID_RADII[size]
		self.direction = random_obj.uniform(0, 2 * pi) if direction is None else direction
		vel_unit_vector = vec2(cos(self.direction), sin(self.direction))
		vel_magnitude = random_obj.uniform(ASTEROID_VELS[size] * 0.2, ASTEROID_VELS[size])
		self.vel = vel_unit_vector * vel_magnitude

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


class Bullet:
	def __init__(self, pos, direction):
		self.pos = pos
		self.last_pos = pos
		vel_unit_vector = vec2(cos(direction), sin(direction))
		self.vel = vel_unit_vector * BULLET_SPEED
		self.life = BULLET_LIFESPAN

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
		bullet_path = (self.pos, self.last_pos)
		for a1, a2 in asteroid.lines:
			if lines_intersect(bullet_path, (a1, a2)):
				return True
		return False


class GameEnv:
	def __init__(self, *, random_obj, do_rendering, training_mode=False):
		self.rand = random_obj
		self.training_mode = training_mode
		self.spaceship = None
		self.asteroids = None
		self.level = None
		self.timestep_reward = None

		self.detected_asteroid_idx = None  # Indices of the nearest MAX_ASTEROIDS_DETECT asteroids
		self.nearest_asteroid_idx = None   # Index of the nearest asteroid (for predictive aiming)

		if do_rendering:
			pg.init()
			pg.display.set_caption('PPO Asteroids player')
			self.font = pg.font.SysFont('lucidasanstypewriteroblique', 26)
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
			distance_to_centre = (dx * dx + dy * dy) ** 0.5
			distance = distance_to_centre \
				- (asteroid.radius + (min(asteroid.rand_offsets) + max(asteroid.rand_offsets)) / 2) \
				- SPACESHIP_SCALE

			return distance

		def toroidal_direction_to_asteroid(asteroid, spaceship_direction):
			"""Toroidal direction from the spaceship to a given asteroid"""

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

			# Find the nearest position considering wrap-around
			nearest_pos = min(possible_positions, key=lambda pos: pos[0] * pos[0] + pos[1] * pos[1])

			# Calculate the angle to the nearest position
			angle_to_asteroid = atan2(nearest_pos[1], nearest_pos[0])

			# -pi to pi
			direction_to_asteroid = (angle_to_asteroid - spaceship_direction + pi / 2) % (2 * pi)
			if direction_to_asteroid > pi:
				direction_to_asteroid -= 2 * pi

			return direction_to_asteroid

		def find_aim_angle(gun_pos, asteroid, spaceship_heading):
			"""Given the spaceship gun and a moving target (asteroid), find the angle to aim at in order to hit it"""

			def wrap_around_delta_pos(pos1, pos2):
				dx = pos2[0] - pos1[0]
				dy = pos2[1] - pos1[1]
				dx_wrap = dx if abs(dx) < SCENE_WIDTH / 2 else dx - np.sign(dx) * SCENE_WIDTH
				dy_wrap = dy if abs(dy) < SCENE_HEIGHT / 2 else dy - np.sign(dy) * SCENE_HEIGHT

				return np.array([dx_wrap, dy_wrap])

			def equation(t):
				future_target_pos = asteroid.pos + asteroid.vel * t
				# Calculate relative position considering wrap-around
				rel_pos = wrap_around_delta_pos(gun_pos, future_target_pos)
				# Distance to cover by bullet
				dist = np.linalg.norm(rel_pos)

				# Solving: dist - BULLET_SPEED * t = 0 (i.e. solve for t)
				return dist - BULLET_SPEED * t


			# Solve for time of impact (initial guess: t = 1)
			t_impact = fsolve(equation, x0=1)[0]

			# Calculate the relative position at impact time
			impact_pos = wrap_around_delta_pos(gun_pos, asteroid.pos + asteroid.vel * t_impact)

			# Find the aim angle (-pi to pi)
			angle = (atan2(impact_pos[1], impact_pos[0]) - spaceship_heading + pi / 2) % (2 * pi)
			if angle > pi:
				angle -= 2 * pi

			return angle


		# 1. Caluclate distances to all asteroids

		dists_to_asteroids = np.array([toroidal_distance_to_asteroid(a) for a in self.asteroids])

		# 2. Get asteroid indices based on distance

		self.nearest_asteroid_idx = np.argmin(dists_to_asteroids)

		if self.detected_asteroid_idx is None or max(self.detected_asteroid_idx) >= len(self.asteroids):
			self.detected_asteroid_idx = np.argsort(dists_to_asteroids)[:MAX_ASTEROIDS_DETECT]
		dists_to_asteroids = dists_to_asteroids[self.detected_asteroid_idx]
		pad_size = max(0, MAX_ASTEROIDS_DETECT - len(self.detected_asteroid_idx))

		# 3. Find the following asteroid info

		directions_to_asteroids = []
		rel_asteroid_vel_mags = []
		rel_asteroid_vel_angles = []

		# Convert spaceship heading and direction to [-pi, pi] range
		adjusted_heading = (self.spaceship.heading + pi / 2) % (2 * pi)
		if adjusted_heading > pi:
			adjusted_heading -= 2 * pi
		spaceship_direction = atan2(self.spaceship.vel.y, self.spaceship.vel.x)
		adjusted_direction = (spaceship_direction + pi / 2) % (2 * pi)
		if adjusted_direction > pi:
			adjusted_direction -= 2 * pi

		for idx in self.detected_asteroid_idx:
			asteroid = self.asteroids[idx]
			direction_to_asteroid = toroidal_direction_to_asteroid(asteroid, adjusted_direction)
			mag_diff = asteroid.vel.magnitude() - self.spaceship.vel.magnitude()
			angle_diff = asteroid.direction - spaceship_direction
			if angle_diff > pi:
				angle_diff -= 2 * pi

			directions_to_asteroids.append(direction_to_asteroid)
			rel_asteroid_vel_mags.append(mag_diff)
			rel_asteroid_vel_angles.append(angle_diff)

		# 4. Find angle to aim at in order to hit the nearest asteroid

		gun_pos = vec2(self.spaceship.lines[0][0])
		nearest_asteroid = self.asteroids[self.nearest_asteroid_idx]
		aim_angle = find_aim_angle(gun_pos, nearest_asteroid, adjusted_heading)

		# 5. The further the spaceship is from asteroids, the greater the timestep reward

		dists_to_asteroids /= MAX_ASTEROID_DIST
		nearest_3_dists = dists_to_asteroids[:3]
		self.timestep_reward = 2 * sum(nearest_3_dists) / len(nearest_3_dists)

		# 6. Construct state representation

		dists_to_asteroids = np.pad(dists_to_asteroids, (0, pad_size), constant_values=(1,))
		directions_to_asteroids = np.pad(directions_to_asteroids, (0, pad_size))
		rel_asteroid_vel_mags = np.pad(rel_asteroid_vel_mags, (0, pad_size))
		rel_asteroid_vel_angles = np.pad(rel_asteroid_vel_angles, (0, pad_size))

		state = [
			aim_angle / pi,
			*dists_to_asteroids,
			*directions_to_asteroids / pi,
			*rel_asteroid_vel_mags / MAX_VEL,
			*rel_asteroid_vel_angles / pi
		]

		return state

	def step(self, action):
		"""
		Perform an action, then return resulting info as the tuple (return, next_state, terminal)
		"""

		self.spaceship.perform_action(action)

		# Update asteroids and bullets
		for a in self.asteroids:
			a.update()
		for b in self.spaceship.bullets:
			b.update()
			if b.life == 0:
				self.spaceship.bullets.remove(b)

		next_state = self.get_state()

		# Check for spaceship-asteroid collision
		if any(self.spaceship.check_collision(a) for a in self.asteroids):
			return COLLISION_PENALTY, next_state, True

		# Check if a bullet has hit an asteroid
		for bullet in self.spaceship.bullets:
			for idx, asteroid in enumerate(self.asteroids):
				if bullet.check_asteroid_hit(asteroid):
					self.asteroids.remove(asteroid)
					self.spaceship.bullets.remove(bullet)
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

					self.detected_asteroid_idx = None

					if len(self.asteroids) == 0:  # Destroyed all asteroids, go to next level
						self.level += 1
						self.spaceship.bullets = []
						self.make_asteroids()

					if idx == self.nearest_asteroid_idx:
						return AGENT_ASTEROID_DESTROY_REWARD + self.timestep_reward, next_state, False
					else:
						return self.timestep_reward, next_state, False

		return self.timestep_reward, next_state, False

	def render(self, action, terminal):
		self.scene.fill('black')

		# Draw spaceship
		flame_green = np.random.randint(0, 256)  # Using self.rand gives non-reproducible testing results
		for idx, (s1, s2) in enumerate(self.spaceship.lines):
			if idx < 3:
				# Hull = red if game over, else white
				colour = (255, 0, 0) if terminal else (255, 255, 255)
			else:
				# Thruster = red/orange/yellow
				colour = (255, flame_green, 0)
			pg.draw.line(self.scene, colour, s1, s2, 2)

		# Draw asteroids
		for a in self.asteroids:
			for a1, a2 in a.lines:
				pg.draw.line(self.scene, (255, 255, 255), a1, a2)

		# Draw bullets
		for b in self.spaceship.bullets:
			pg.draw.circle(self.scene, (255, 255, 255), b.pos, 3)

		# Display arrow/space key control (key is green if used, else grey)
		up_colour = (0, 255, 0) if action in (1, 5, 6, 7, 10, 11) else (80, 80, 80)
		left_colour = (0, 255, 0) if action in (2, 5, 8, 10) else (80, 80, 80)
		right_colour = (0, 255, 0) if action in (3, 6, 9, 11) else (80, 80, 80)
		space_colour = (0, 255, 0) if action in (4, 7, 8, 9, 10, 11) else (80, 80, 80)
		pg.draw.rect(self.scene, up_colour, pg.Rect(120, 80, 25, 25))
		pg.draw.rect(self.scene, left_colour, pg.Rect(95, 105, 25, 25))
		pg.draw.rect(self.scene, right_colour, pg.Rect(145, 105, 25, 25))
		pg.draw.rect(self.scene, space_colour, pg.Rect(12, 105, 75, 25))

		# Display level and score
		level_lbl = self.font.render(f'Level: {self.level}', True, (255, 255, 255))
		score_lbl = self.font.render(f'Score: {self.spaceship.score}', True, (255, 255, 255))
		self.scene.blit(level_lbl, (10, 10))
		self.scene.blit(score_lbl, (10, 40))

		pg.display.update()
		if terminal:
			pg.time.wait(800)
		self.clock.tick(FPS)

	def make_asteroids(self):
		"""Num. asteroids = level + 3"""

		# Don't spawn asteroids around where the spaceship is (also considering wrap-around)
		possible_x = [x for x in range(SCENE_WIDTH) if min(abs(x - self.spaceship.pos.x), SCENE_WIDTH - abs(x - self.spaceship.pos.x)) > 250]
		possible_y = [y for y in range(SCENE_HEIGHT) if min(abs(y - self.spaceship.pos.y), SCENE_HEIGHT - abs(y - self.spaceship.pos.y)) > 250]

		self.asteroids = []
		for i in range(self.level + 3):
			match i % 4:
				case 0: asteroid = Asteroid(self.rand, 'large', vec2(y=self.rand.choice(possible_y)))
				case 1: asteroid = Asteroid(self.rand, 'large', vec2(SCENE_WIDTH - 1, self.rand.choice(possible_y)))
				case 2: asteroid = Asteroid(self.rand, 'large', vec2(self.rand.choice(possible_x), 0))
				case _: asteroid = Asteroid(self.rand, 'large', vec2(self.rand.choice(possible_x), SCENE_HEIGHT - 1))

			self.asteroids.append(asteroid)

	def reset(self, seed=None):
		if seed is not None:
			self.rand.seed(seed)

		self.spaceship = Spaceship()

		# If training, initialise spaceship with a random position/orientation/velocity each time
		# (more variation in initial states)
		if self.training_mode:
			self.spaceship.pos = vec2(self.rand.randrange(SCENE_WIDTH), self.rand.randrange(SCENE_HEIGHT))
			self.spaceship.heading = self.rand.uniform(0, 2 * pi)
			rand_direction = self.rand.uniform(0, 2 * pi)
			rand_magnitude = self.rand.uniform(0, MAX_VEL / 2)
			self.spaceship.vel = vec2(cos(rand_direction), sin(rand_direction)) * rand_magnitude

		self.level = 1
		self.make_asteroids()
