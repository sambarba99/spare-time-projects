# Boid class for flockingSimulator.py
# Author: Sam Barba
# Created 11/03/2022

import numpy as np

class Boid:
    def __init__(self, perception_radius, max_steering_force, max_x, max_y):
        self.perception_radius = perception_radius
        self.max_steering_force = max_steering_force
        self.max_vel = 8
        self.max_x = max_x
        self.max_y = max_y
        self.pos = np.array([np.random.uniform(max_x), np.random.uniform(max_y)])
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
        if self.pos[0] > self.max_x: self.pos[0] = 0
        if self.pos[0] < 0: self.pos[0] = self.max_x
        if self.pos[1] > self.max_y: self.pos[1] = 0
        if self.pos[1] < 0: self.pos[1] = self.max_y

    # Steer to avoid crowding boids in local flock
    def separation(self, flock):
        steering = np.zeros(2)
        total = 0

        for boid in flock:
            if self is boid: continue

            d = np.linalg.norm(self.pos - boid.pos)
            if d < self.perception_radius:
                steering += (self.pos - boid.pos) / (d + 0.001) ** 2  # Avoid 0 div error
                total += 1

        if total > 0:
            steering /= total
            # Steer at max velocity
            steering = self.__scale_vector(steering, self.max_vel)

        return steering

    # Steer towards the average heading of boids in local flock
    def alignment(self, flock):
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

    # Steer towards the average position of boids in local flock
    def cohesion(self, flock):
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

    # Limit a vector magnitude whilst preserving direction (e.g. steering force or velocity)
    def __limit_vector(self, vector, limit):
        magnitude = np.linalg.norm(vector)
        return vector * limit / magnitude if magnitude > limit else vector

    # Scale a vector magnitude whilst preserving direction
    def __scale_vector(self, vector, scale):
        magnitude = np.linalg.norm(vector)
        unit_vector = vector / magnitude
        return unit_vector * scale
