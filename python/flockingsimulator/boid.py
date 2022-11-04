"""
Boid class for flocking_simulator.py

Author: Sam Barba
Created 11/03/2022
"""

import numpy as np

class Boid:
	def __init__(self, perception_radius, max_steering_force, x_max, y_max):
		self.perception_radius = perception_radius
		self.max_steering_force = max_steering_force
		self.max_vel = 8
		self.x_max = x_max
		self.y_max = y_max
		self.pos = np.array([np.random.uniform(x_max), np.random.uniform(y_max)])
		self.vel = np.random.uniform(-self.max_vel, self.max_vel, size=2)
		self.acc = np.zeros(2)

	def apply_behaviour(self, flock, do_separation, do_alignment, do_cohesion):
		self.acc *= 0  # Reset to 0 net acceleration

		# Let separation be 1% stronger than other forces, so boids fill screen more
		if do_separation: self.acc += self.separation(flock) * 1.01
		if do_alignment: self.acc += self.alignment(flock)
		if do_cohesion: self.acc += self.cohesion(flock)

		self.__limit_vector(self.acc, self.max_steering_force)

	def update(self):
		self.vel += self.acc
		self.vel = self.__limit_vector(self.vel, self.max_vel)
		self.pos += self.vel

		# Check bounds
		if self.pos[0] > self.x_max: self.pos[0] = 0
		if self.pos[0] < 0: self.pos[0] = self.x_max
		if self.pos[1] > self.y_max: self.pos[1] = 0
		if self.pos[1] < 0: self.pos[1] = self.y_max

	def separation(self, flock):
		"""Steer to avoid crowding boids in local flock"""

		steering = np.zeros(2)
		total = 0

		for boid in flock:
			if self is boid: continue

			d = np.linalg.norm(self.pos - boid.pos)
			if d < self.perception_radius:
				steering += (self.pos - boid.pos) / (d + 1e-6) ** 2  # Avoid 0 div error
				total += 1

		if total > 0:
			steering /= total
			# Steer at max velocity
			steering = self.__scale_vector(steering, self.max_vel)

		return steering

	def alignment(self, flock):
		"""Steer towards the average heading of boids in local flock"""

		steering = np.zeros(2)
		total = 0

		for boid in flock:
			if self is boid: continue

			d = np.linalg.norm(self.pos - boid.pos)
			if d < self.perception_radius:
				steering += boid.vel
				total += 1

		if total > 0:
			steering /= total
			steering = self.__scale_vector(steering, self.max_vel)

		return steering

	def cohesion(self, flock):
		"""Steer towards the average position of boids in local flock"""

		steering = np.zeros(2)
		total = 0

		for boid in flock:
			if self is boid: continue

			d = np.linalg.norm(self.pos - boid.pos)
			if d < self.perception_radius:
				steering += boid.pos
				total += 1

		if total > 0:
			steering /= total
			steering -= self.pos
			steering = self.__scale_vector(steering, self.max_vel)

		return steering

	def __limit_vector(self, vector, limit):
		"""Limit a vector magnitude whilst preserving direction (e.g. steering force or velocity)"""
		magnitude = np.linalg.norm(vector)
		return vector * limit / magnitude if magnitude > limit else vector

	def __scale_vector(self, vector, scale):
		"""Scale a vector magnitude whilst preserving direction"""
		magnitude = np.linalg.norm(vector)
		unit_vector = vector / magnitude
		return unit_vector * scale
