"""
Boid class

Author: Sam Barba
Created 11/03/2022
"""

import numpy as np
import pygame as pg


vec2 = pg.math.Vector2


class Boid:
	def __init__(self, perception_radius, max_force, x_max, y_max):
		self.perception_radius = perception_radius
		self.max_force = max_force
		self.max_vel = 8
		self.x_max = x_max
		self.y_max = y_max
		self.pos = vec2(np.random.uniform(x_max), np.random.uniform(y_max))
		self.vel = vec2(*np.random.uniform(-self.max_vel, self.max_vel, size=2))
		self.acc = vec2()


	def apply_behaviour(self, flock, do_separation, do_alignment, do_cohesion):
		self.acc *= 0  # Reset to 0 net acceleration

		# Let separation be 1% stronger than other forces, so boids fill screen more
		if do_separation: self.acc += self.separation(flock) * 1.01
		if do_alignment: self.acc += self.alignment(flock)
		if do_cohesion: self.acc += self.cohesion(flock)

		try:
			self.acc = self.acc.clamp_magnitude(self.max_force)
		except ValueError:
			pass  # In case of 'ValueError: Cannot clamp a vector with zero length'


	def update(self):
		self.vel += self.acc
		self.vel = self.vel.clamp_magnitude(self.max_force)
		self.pos += self.vel

		# Check bounds
		if self.pos[0] > self.x_max: self.pos[0] = 0
		if self.pos[0] < 0: self.pos[0] = self.x_max
		if self.pos[1] > self.y_max: self.pos[1] = 0
		if self.pos[1] < 0: self.pos[1] = self.y_max


	def separation(self, flock):
		"""Steer to avoid crowding in local flock"""

		steering = vec2()
		total = 0

		for boid in flock:
			if self is boid: continue

			d = self.pos.distance_to(boid.pos)
			if d < self.perception_radius:
				steering += (self.pos - boid.pos) / (d + 1e-6) ** 2  # Avoid 0 div error
				total += 1

		if total:
			steering /= total
			# Steer at max velocity
			try:
				steering.scale_to_length(self.max_vel)
			except ValueError:
				pass  # In case of 'ValueError: Cannot scale a vector with zero length'

		return steering


	def alignment(self, flock):
		"""Steer towards the average heading of local flock"""

		steering = vec2()
		total = 0

		for boid in flock:
			if self is boid: continue

			if self.pos.distance_to(boid.pos) < self.perception_radius:
				steering += boid.vel
				total += 1

		if total:
			steering /= total
			try:
				steering.scale_to_length(self.max_vel)
			except ValueError:
				pass

		return steering


	def cohesion(self, flock):
		"""Steer towards the average position of local flock"""

		steering = vec2()
		total = 0

		for boid in flock:
			if self is boid: continue

			if self.pos.distance_to(boid.pos) < self.perception_radius:
				steering += boid.pos
				total += 1

		if total:
			steering /= total
			steering -= self.pos
			try:
				steering.scale_to_length(self.max_vel)
			except ValueError:
				pass

		return steering
