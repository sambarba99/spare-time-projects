"""
Game environment functionality

Author: Sam Barba
Created 2024-07-11
"""

from math import pi, sin, cos, atan2

import numpy as np
import pygame as pg

from proximal_policy_optimisation_asteroids.game_env.constants import *
from proximal_policy_optimisation_asteroids.ppo.constants import MAX_ASTEROIDS_DETECT


# Normalised centered asteroid points
ASTEROIDS = [
	{
		'points': [(-2, -4), (0, -2), (2, -4), (4, -2), (3, 0), (4, 2), (1, 4), (-2, 4), (-4, 2), (-4, -2)],
		'mean_radius': 4.04
	},
	{
		'points': [(-2, -4), (0, -3), (2, -4), (4, -2), (2, -1), (4, 1), (2, 4), (-1, 3), (-2, 4), (-4, 2), (-3, 0), (-4, -2)],
		'mean_radius': 3.90
	},
	{
		'points': [(-1, -4), (2, -4), (4, -1), (4, 1), (2, 4), (0, 4), (0, 1), (-2, 4), (-4, 1), (-2, 0), (-4, -1)],
		'mean_radius': 3.73
	},
	{
		'points': [(-2, -4), (1, -4), (4, -2), (4, -1), (1, 0), (4, 2), (2, 4), (1, 3), (-2, 4), (-4, 1), (-4, -2), (-1, -2)],
		'mean_radius': 3.80
	}
]

# For normalising radii in state representation
MAX_ASTEROID_RADIUS = max(a['mean_radius'] for a in ASTEROIDS) * ASTEROID_SCALES['large']

# Normalised centered spaceship points
SHIP_POINTS = [(0, -11), (-6, 7), (6, 7), (-5, 4), (5, 4), (-3, 4), (3, 4), (0, 10)]

# Indices connecting points to form lines
SHIP_EDGES = [
	(0, 1), (0, 2), (3, 4),  # Hull lines
	(5, 7), (6, 7)  # Thruster lines
]

vec2 = pg.math.Vector2


def normalise_angle(a):
	return (a + pi) % (2 * pi) - pi


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
		self.heading = 0  # 0 = point up
		self.bullets = []
		self.score = 0

		# Lines defining the spaceship (hull lines + thruster lines)
		self.lines = None
		self.update_lines()

	def perform_action(self, action):
		"""Perform an action e.g. boosting, and update spaceship properties accordingly"""

		# Decode action num

		boosting = action in (1, 5, 6, 7, 10, 11)
		turning_left = action in (2, 5, 8, 10)
		turning_right = action in (3, 6, 9, 11)
		shooting = action in (4, 7, 8, 9, 10, 11)

		# Apply force to accelerate/decelerate if necessary

		self.acc = ACCELERATION_FORCE if boosting else 0

		# Apply steering if necessary

		if turning_left:
			self.heading = normalise_angle(self.heading - TURN_RATE)
		elif turning_right:
			self.heading = normalise_angle(self.heading + TURN_RATE)

		# Update velocity

		if self.acc:
			acc_vector = vec2(sin(self.heading), -cos(self.heading)) * self.acc
			self.vel += acc_vector
			self.vel.clamp_magnitude_ip(MAX_VEL)

		# Update position

		self.pos += self.vel
		self.pos.x %= SCENE_WIDTH
		self.pos.y %= SCENE_HEIGHT
		self.update_lines()

		# Add bullets

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

		return self.pos.distance_to(asteroid.pos) < asteroid.radius

	def update_lines(self):
		sin_heading = sin(self.heading)
		cos_heading = cos(self.heading)
		transformed_points = []

		for x, y in SHIP_POINTS:
			rx = (x * cos_heading - y * sin_heading) * SPACESHIP_SCALE
			ry = (x * sin_heading + y * cos_heading) * SPACESHIP_SCALE
			transformed_points.append((rx + self.pos.x, ry + self.pos.y))

		# Hull lines
		self.lines = [
			(transformed_points[start], transformed_points[end])
			for start, end in SHIP_EDGES[:3]
		]

		if self.acc:
			# Thruster lines
			self.lines.extend([
				(transformed_points[start], transformed_points[end])
				for start, end in SHIP_EDGES[3:]
			])


class Asteroid:
	def __init__(self, random_obj, size, pos, direction=None):
		self.size = size
		self.pos = pos
		self.direction = random_obj.uniform(0, 2 * pi) if direction is None else direction
		vel_unit_vector = vec2(sin(self.direction), -cos(self.direction))
		vel_magnitude = random_obj.uniform(ASTEROID_VELS[size] * 0.5, ASTEROID_VELS[size])
		self.vel = vel_unit_vector * vel_magnitude

		rand_asteroid = random_obj.choice(ASTEROIDS)
		self.scaled_points = [vec2(x, y) * ASTEROID_SCALES[size] for x, y in rand_asteroid['points']]
		self.radius = rand_asteroid['mean_radius'] * ASTEROID_SCALES[size]

		# Lines defining the asteroid
		self.lines = None
		self.update()

	def update(self):
		self.pos += self.vel
		self.pos.x %= SCENE_WIDTH
		self.pos.y %= SCENE_HEIGHT

		local_points = [
			(p.x + self.pos.x, p.y + self.pos.y)
			for p in self.scaled_points
		]

		self.lines = [
			(local_points[i], local_points[(i + 1) % len(local_points)])
			for i in range(len(local_points))
		]


class Bullet:
	def __init__(self, pos, direction):
		self.pos = pos
		vel_unit_vector = vec2(sin(direction), -cos(direction))
		self.vel = vel_unit_vector * BULLET_SPEED
		self.life = BULLET_LIFESPAN

	def update(self):
		self.pos += self.vel
		self.pos.x %= SCENE_WIDTH
		self.pos.y %= SCENE_HEIGHT
		self.life -= 1

	def check_asteroid_hit(self, asteroid):
		return self.pos.distance_to(asteroid.pos) < asteroid.radius


class GameEnv:
	def __init__(self, *, random_obj, do_rendering, training_mode=False):
		self.rand = random_obj
		self.training_mode = training_mode
		self.spaceship = None
		self.asteroids = None
		self.level = None
		self.detected_asteroids = None

		if do_rendering:
			pg.init()
			pg.display.set_caption('PPO Asteroids player')
			self.player = 'you'
			self.font28 = pg.font.SysFont('consolas', 28)
			self.font22 = pg.font.SysFont('consolas', 22)
			self.scene = pg.display.set_mode((SCENE_WIDTH, SCENE_HEIGHT))
			self.clock = pg.time.Clock()

		self.reset()

	def get_state(self, return_timestep_reward=False):
		"""
		Obtain the state of the environment by engineering features about the spaceship (inc. bullets) and asteroids
		"""

		def toroidal_relative_vector(a, b):
			"""Smallest toroidal displacement from point A to point B"""

			dx = (b.x - a.x + SCENE_WIDTH / 2) % SCENE_WIDTH - SCENE_WIDTH / 2
			dy = (b.y - a.y + SCENE_HEIGHT / 2) % SCENE_HEIGHT - SCENE_HEIGHT / 2

			return vec2(dx, dy)


		# Spaceship info

		spaceship_vel_mag = self.spaceship.vel.magnitude() / MAX_VEL
		if spaceship_vel_mag > 0:
			spaceship_vel_dir = atan2(self.spaceship.vel.x, -self.spaceship.vel.y)
			spaceship_vel_sin = sin(spaceship_vel_dir)
			spaceship_vel_cos = cos(spaceship_vel_dir)
		else:
			spaceship_vel_sin = spaceship_vel_cos = 0

		# Asteroid info

		asteroid_info = []

		for idx, a in enumerate(self.asteroids):
			rel_pos = toroidal_relative_vector(self.spaceship.pos, a.pos)
			distance = rel_pos.magnitude()
			rel_direction = normalise_angle(atan2(rel_pos.x, -rel_pos.y) - self.spaceship.heading)
			rel_vel = a.vel - self.spaceship.vel

			asteroid_info.append({
				'idx': idx,
				'dist': distance,
				'rel_direction': rel_direction,
				'rel_vel': rel_vel,
				'radius': a.radius
			})

		# Bullet info

		bullet_dists = []
		rel_bullet_directions = []
		bullet_life_left = []

		for b in self.spaceship.bullets:
			rel_pos = toroidal_relative_vector(self.spaceship.pos, b.pos)
			distance = rel_pos.magnitude()
			rel_direction = normalise_angle(atan2(rel_pos.x, -rel_pos.y) - self.spaceship.heading)

			bullet_dists.append(distance)
			rel_bullet_directions.append(rel_direction)
			bullet_life_left.append(b.life)

		# Prioritise detected asteroids based on sorted distance and direction (to minimise shuffling between frames),
		# keeping top MAX_ASTEROIDS_DETECT

		asteroid_info.sort(key=lambda i: (i['dist'], i['rel_direction']))
		detected_asteroids = asteroid_info[:MAX_ASTEROIDS_DETECT]
		if self.training_mode:
			self.detected_asteroids = {a['idx'] for a in detected_asteroids}

		# Extract features for top asteroids

		asteroid_dists = [a['dist'] for a in detected_asteroids]
		rel_asteroid_directions = [a['rel_direction'] for a in detected_asteroids]
		rel_asteroid_vel_mags = [a['rel_vel'].magnitude() for a in detected_asteroids]
		rel_asteroid_vel_directions = [
			atan2(a['rel_vel'].x, -a['rel_vel'].y) - self.spaceship.heading
			for a in detected_asteroids
		]
		asteroid_radii = [a['radius'] for a in detected_asteroids]

		# Normalise observations to [0,1] or [-1,1]

		asteroid_dists = np.array(asteroid_dists) / MAX_OBJECT_DIST
		rel_asteroid_directions_sin = np.sin(rel_asteroid_directions)
		rel_asteroid_directions_cos = np.cos(rel_asteroid_directions)
		rel_asteroid_vel_mags = np.array(rel_asteroid_vel_mags) / (MAX_VEL + ASTEROID_VELS['small'])
		rel_asteroid_vel_sin = np.sin(rel_asteroid_vel_directions)
		rel_asteroid_vel_cos = np.cos(rel_asteroid_vel_directions)
		asteroid_radii = np.array(asteroid_radii) / MAX_ASTEROID_RADIUS
		bullet_dists = np.array(bullet_dists) / MAX_OBJECT_DIST
		rel_bullet_directions_sin = np.sin(rel_bullet_directions)
		rel_bullet_directions_cos = np.cos(rel_bullet_directions)
		bullet_life_left = np.array(bullet_life_left) / BULLET_LIFESPAN

		# Apply padding so that the final state vector has a consistent shape

		pad_size_asteroid = max(0, MAX_ASTEROIDS_DETECT - len(detected_asteroids))
		pad_size_bullet = max(0, MAX_BULLETS - len(bullet_dists))

		if pad_size_asteroid:
			asteroid_dists = np.pad(asteroid_dists, (0, pad_size_asteroid))
			rel_asteroid_directions_sin = np.pad(rel_asteroid_directions_sin, (0, pad_size_asteroid))
			rel_asteroid_directions_cos = np.pad(rel_asteroid_directions_cos, (0, pad_size_asteroid))
			rel_asteroid_vel_mags = np.pad(rel_asteroid_vel_mags, (0, pad_size_asteroid))
			rel_asteroid_vel_sin = np.pad(rel_asteroid_vel_sin, (0, pad_size_asteroid))
			rel_asteroid_vel_cos = np.pad(rel_asteroid_vel_cos, (0, pad_size_asteroid))
			asteroid_radii = np.pad(asteroid_radii, (0, pad_size_asteroid))
		asteroid_mask = np.pad(np.ones(len(detected_asteroids)), (0, pad_size_asteroid))

		if pad_size_bullet:
			bullet_dists = np.pad(bullet_dists, (0, pad_size_bullet))
			rel_bullet_directions_sin = np.pad(rel_bullet_directions_sin, (0, pad_size_bullet))
			rel_bullet_directions_cos = np.pad(rel_bullet_directions_cos, (0, pad_size_bullet))
			bullet_life_left = np.pad(bullet_life_left, (0, pad_size_bullet), constant_values=1)

		# Construct state representation

		asteroid_info_stacked = np.stack([
			asteroid_dists,
			rel_asteroid_directions_sin,
			rel_asteroid_directions_cos,
			rel_asteroid_vel_mags,
			rel_asteroid_vel_sin,
			rel_asteroid_vel_cos,
			asteroid_radii
		], axis=1)
		asteroid_info_flattened = asteroid_info_stacked.flatten()

		bullet_info_stacked = np.stack([
			bullet_dists,
			rel_bullet_directions_sin,
			rel_bullet_directions_cos,
			bullet_life_left
		], axis=1)
		bullet_info_flattened = bullet_info_stacked.flatten()

		state = [
			# Spaceship info
			spaceship_vel_mag,
			spaceship_vel_sin,
			spaceship_vel_cos,
			sin(self.spaceship.heading),
			cos(self.spaceship.heading),
			*bullet_info_flattened,
			# Asteroid info
			*asteroid_info_flattened,
			*asteroid_mask
		]

		if return_timestep_reward:
			# The further the spaceship is from the nearest asteroid, the greater the timestep reward
			timestep_reward = asteroid_dists[0]

			return state, timestep_reward
		else:
			return state

	def step(self, action):
		"""
		Perform an action, then return resulting info as the tuple (reward, next_state, terminal)
		"""

		self.spaceship.perform_action(action)

		# Update asteroids and bullets

		for a in self.asteroids:
			a.update()

		total_miss_penalty = 0

		for b in self.spaceship.bullets[:]:
			b.update()
			if b.life == 0:
				self.spaceship.bullets.remove(b)
				total_miss_penalty += MISS_PENALTY

		if self.training_mode:
			prev_detected = self.detected_asteroids.copy()
			next_state, timestep_reward = self.get_state(return_timestep_reward=True)
		else:
			prev_detected = set()
			next_state = self.get_state()
			timestep_reward = 0

		timestep_reward += total_miss_penalty

		# Check for spaceship-asteroid collision

		if any(self.spaceship.check_collision(a) for a in self.asteroids):
			return COLLISION_PENALTY, next_state, True

		# Check if a bullet has hit an asteroid

		for bullet in self.spaceship.bullets[:]:
			for idx, asteroid in enumerate(self.asteroids[:]):
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
									asteroid.direction + self.rand.uniform(-pi / 8, pi / 8)
								)
							)

					if len(self.asteroids) == 0:  # Destroyed all asteroids, go to next level
						self.level += 1
						self.spaceship.bullets = []
						self.make_asteroids()

					if idx in prev_detected:
						timestep_reward += AGENT_ASTEROID_DESTROY_REWARD

					return timestep_reward, next_state, False

		return timestep_reward, next_state, False

	def render(self, action, terminal):
		self.scene.fill('black')

		# Render spaceship

		is_full_ship = len(self.spaceship.lines) == len(SHIP_EDGES)
		green = np.random.randint(0, 256)  # Using self.rand gives non-reproducible testing results
		hull_colour = 'red' if terminal else 'white'

		# Reversed so thruster lines render underneath hull lines
		for idx, (start_point, end_point) in enumerate(reversed(self.spaceship.lines)):
			is_thruster = is_full_ship and idx < 2
			colour = (255, green, 0) if is_thruster else hull_colour
			pg.draw.line(self.scene, colour, start_point, end_point, 3)

		# Render asteroids and bullets

		for a in self.asteroids:
			for a1, a2 in a.lines:
				pg.draw.line(self.scene, 'white', a1, a2, 2)

		for b in self.spaceship.bullets:
			pg.draw.circle(self.scene, 'white', b.pos, 3)

		# Render arrow/space key control (key is green if used, else grey)

		if terminal:
			w_colour = a_colour = d_colour = space_colour = 'red'
		else:
			w_colour = 'green' if action in (1, 5, 6, 7, 10, 11) else (40, 40, 40)
			a_colour = 'green' if action in (2, 5, 8, 10) else (40, 40, 40)
			d_colour = 'green' if action in (3, 6, 9, 11) else (40, 40, 40)
			space_colour = 'green' if action in (4, 7, 8, 9, 10, 11) else (40, 40, 40)
		pg.draw.rect(self.scene, w_colour, pg.Rect(56, 25, 30, 30))
		pg.draw.rect(self.scene, a_colour, pg.Rect(25, 56, 30, 30))
		pg.draw.rect(self.scene, d_colour, pg.Rect(87, 56, 30, 30))
		pg.draw.rect(self.scene, space_colour, pg.Rect(120, 56, 120, 30))

		# Render level, score, player labels

		level_lbl = self.font28.render(f'Level: {self.level}', True, 'red' if terminal else 'white')
		score_lbl = self.font28.render(f'Score: {self.spaceship.score}', True, 'red' if terminal else 'white')
		player_lbl = self.font22.render(f'Player: {self.player}', True, 'red' if terminal else 'white')
		self.scene.blit(level_lbl, (24, 104))
		self.scene.blit(score_lbl, (24, 138))
		self.scene.blit(player_lbl, (24, 173))

		pg.display.update()
		if terminal:
			pg.time.wait(1000)
		self.clock.tick(FPS)

	def make_asteroids(self):
		"""No. asteroids = level + 3"""

		# Don't spawn asteroids around where the spaceship is (also considering wrap-around)
		possible_x = [
			x for x in range(SCENE_WIDTH)
			if min(abs(x - self.spaceship.pos.x), SCENE_WIDTH - abs(x - self.spaceship.pos.x)) > 250
		]
		possible_y = [
			y for y in range(SCENE_HEIGHT)
			if min(abs(y - self.spaceship.pos.y), SCENE_HEIGHT - abs(y - self.spaceship.pos.y)) > 250
		]

		self.asteroids = []
		for _ in range(self.level + 3):
			edge = self.rand.randrange(4)
			match edge:
				case 0: asteroid = Asteroid(self.rand, 'large', vec2(0, self.rand.choice(possible_y)))
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
			self.spaceship.vel = vec2(sin(rand_direction), -cos(rand_direction)) * rand_magnitude

		self.level = 1
		self.make_asteroids()
