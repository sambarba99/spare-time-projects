"""
Perfectly elastic collision simulator

Author: Sam Barba
Created 28/08/2024
"""

import sys

import numpy as np
import pygame as pg


MIN_MASS = 100
MAX_MASS = 1500
MAX_VEL = 3
NUM_PARTICLES = 100
SCENE_WIDTH = 1200
SCENE_HEIGHT = 800
FPS = 60

vec2 = pg.math.Vector2


class Particle:
	def __init__(self, mass=None, pos=None, vel=None):
		self.mass = np.random.uniform(MIN_MASS, MAX_MASS) if mass is None else mass
		self.radius = self.mass ** 0.5
		self.pos = vec2(
			np.random.uniform(self.radius, SCENE_WIDTH - self.radius),
			np.random.uniform(self.radius, SCENE_HEIGHT - self.radius)
		) if pos is None else pos
		self.vel = vec2(
			np.random.uniform(-MAX_VEL, MAX_VEL),
			np.random.uniform(-MAX_VEL, MAX_VEL)
		) if vel is None else vel

		hue = 60 * (1 - (self.mass - MIN_MASS) / (MAX_MASS - MIN_MASS))
		colour = pg.Color(0)
		colour.hsva = (hue, 100, 100, 100)
		self.colour = (colour.r, colour.g, colour.b)

	def update(self):
		self.pos += self.vel
		if self.pos.x < self.radius:
			self.vel.x *= -1
			self.pos.x = self.radius
		elif self.pos.x > SCENE_WIDTH - self.radius:
			self.vel.x *= -1
			self.pos.x = SCENE_WIDTH - self.radius
		if self.pos.y < self.radius:
			self.vel.y *= -1
			self.pos.y = self.radius
		elif self.pos.y > SCENE_HEIGHT - self.radius:
			self.vel.y *= -1
			self.pos.y = SCENE_HEIGHT - self.radius

	def collide(self, other):
		d = self.pos.distance_to(other.pos) + 2

		if d > self.radius + other.radius:
			return

		impact_vector = other.pos - self.pos

		# Push apart particles so they aren't overlapping
		overlap = d - (self.radius + other.radius)
		delta_pos = impact_vector.copy()
		delta_pos.scale_to_length(overlap * 0.5)
		self.pos += delta_pos
		other.pos -= delta_pos

		# Correct the distance
		d = self.radius + other.radius
		impact_vector.scale_to_length(d)

		# Numerators for updating this particle (a) and other particle (b), and denominator for both
		num_a = (other.vel - self.vel).dot(impact_vector) * 2 * other.mass
		num_b = (other.vel - self.vel).dot(impact_vector) * -2 * self.mass
		den = (self.mass + other.mass) * d * d

		# Update this particle (a)
		delta_v_a = impact_vector.copy()
		delta_v_a *= num_a / den
		self.vel += delta_v_a

		# Update other particle (b)
		delta_v_b = impact_vector.copy()
		delta_v_b *= num_b / den
		other.vel += delta_v_b


def update():
	scene.fill('black')

	for p in particles:
		p.update()

	for i in range(len(particles) - 1):
		for j in range(i + 1, len(particles)):
			particles[i].collide(particles[j])

	for p in particles:
		pg.draw.circle(scene, p.colour, p.pos, p.radius)

	pg.display.update()
	clock.tick(FPS)


if __name__ == '__main__':
	particles = [Particle() for _ in range(NUM_PARTICLES)]

	pg.init()
	pg.display.set_caption('Elastic Collisions')
	scene = pg.display.set_mode((SCENE_WIDTH, SCENE_HEIGHT))
	clock = pg.time.Clock()

	while True:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				sys.exit()

		update()
