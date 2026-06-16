"""
Game environment functionality

Author: Sam Barba
Created 2026-06-05
"""

from math import pi, sin, cos
import random

import pygame as pg

from dueling_ddqn_cartpole_swingup.game_env.constants import *


class CartPole:
	def __init__(self, x_pos, x_vel, angle, angular_vel):
		self.x_pos = x_pos
		self.x_vel = x_vel
		self.angle = angle  # 0 means pole is upright, π means it's hanging down
		self.angular_vel = angular_vel

	def perform_action(self, action):
		"""Perform an action and update the cart-pole accordingly"""

		if action == 0:
			# Do nothing
			force = 0
		else:
			# 1 = push left, 2 = push right
			force = -PUSH_FORCE if action == 1 else PUSH_FORCE

		sin_a = sin(self.angle)
		cos_a = cos(self.angle)

		temp = (force + POLE_MASSLENGTH * self.angular_vel * self.angular_vel * sin_a) / TOTAL_MASS

		angular_acc = (GRAVITY * sin_a - temp * cos_a) / (
			POLE_LENGTH * (4 / 3 - POLE_MASS * cos_a * cos_a / TOTAL_MASS)
		)
		angular_acc -= DAMPING * self.angular_vel

		x_acc = temp - (POLE_MASSLENGTH * angular_acc * cos_a) / TOTAL_MASS
		x_acc -= DAMPING * self.x_vel

		self.x_vel += x_acc * DT
		self.x_pos += self.x_vel * DT

		self.angular_vel += angular_acc * DT
		self.angle += self.angular_vel * DT


class GameEnv:
	def __init__(self, *, do_rendering, training_mode=False):
		self.training_mode = training_mode
		self.max_cart_x = ((SCENE_WIDTH - CART_WIDTH) / 2) / PIXELS_PER_METRE
		self.cart_pole = None

		if do_rendering:
			pg.init()
			pg.display.set_caption('Dueling DDQN Cart-Pole Swing-Up')
			self.player = 'you'
			self.font = pg.font.SysFont('consolas', 20)
			self.scene = pg.display.set_mode((SCENE_WIDTH, SCENE_HEIGHT))
			self.clock = pg.time.Clock()

		self.reset()

	def get_state(self):
		"""Obtain the state of the cart-pole"""

		# Normalise values to [-1,1]

		state = [
			self.cart_pole.x_pos / self.max_cart_x,
			self.cart_pole.x_vel / MAX_VEL,
			sin(self.cart_pole.angle),
			cos(self.cart_pole.angle),
			self.cart_pole.angular_vel / MAX_VEL
		]

		return state

	def step(self, action):
		"""
		Perform an action, then return resulting info as the tuple (reward, next_state, terminal)
		"""

		self.cart_pole.perform_action(action)

		next_state = self.get_state()

		# Check if cart is out of bounds
		cart_centre_px = self.cart_pole.x_pos * PIXELS_PER_METRE
		terminal = abs(cart_centre_px) + CART_WIDTH / 2 > SCENE_WIDTH / 2

		if terminal:
			reward = TERMINAL_PENALTY
		else:
			# Compute weighted reward
			angle_reward = (next_state[-2] + 1) / 2  # Use the cosine of the pole angle
			distance_reward = 1 - (self.cart_pole.x_pos / self.max_cart_x) ** 2
			vel_reward = 1 - (self.cart_pole.x_vel / MAX_VEL) ** 2
			angular_vel_reward = 1 - (self.cart_pole.angular_vel / MAX_VEL) ** 2

			reward = (
				angle_reward * ANGLE_REWARD_SCALE
				+ distance_reward * DISTANCE_REWARD_SCALE
				+ vel_reward * VELOCITY_REWARD_SCALE
				+ angular_vel_reward * ANGULAR_VELOCITY_REWARD_SCALE
			)

		return reward, next_state, terminal

	def render(self, action, terminal):
		self.scene.fill('#a0d0ff')

		# Render track

		pg.draw.line(self.scene, '#606060', (0, TRACK_Y), (SCENE_WIDTH, TRACK_Y), 3)

		# Render cart + wheels

		cart_x = SCENE_WIDTH / 2 + self.cart_pole.x_pos * PIXELS_PER_METRE
		cart_rect = pg.Rect(
			cart_x - CART_WIDTH / 2,
			TRACK_Y - CART_HEIGHT - CART_WHEEL_RADIUS,
			CART_WIDTH,
			CART_HEIGHT
		)

		pg.draw.rect(self.scene, '#805020', cart_rect)
		pg.draw.circle(
			self.scene,
			'black',
			(cart_x - CART_WIDTH / 2, TRACK_Y - CART_WHEEL_RADIUS),
			CART_WHEEL_RADIUS
		)
		pg.draw.circle(
			self.scene,
			'black',
			(cart_x + CART_WIDTH / 2, TRACK_Y - CART_WHEEL_RADIUS),
			CART_WHEEL_RADIUS
		)

		# Render pole and pivot

		pivot_x = cart_x
		pivot_y = cart_rect.top
		pole_length_px = POLE_LENGTH * PIXELS_PER_METRE
		pole_end_x = pivot_x + pole_length_px * sin(self.cart_pole.angle)
		pole_end_y = pivot_y - pole_length_px * cos(self.cart_pole.angle)

		pg.draw.line(self.scene, '#808080', (pivot_x, pivot_y), (pole_end_x, pole_end_y), 6)
		pg.draw.circle(self.scene, '#808080', (pivot_x, pivot_y), 10)

		# Render mass at the end of the pole

		dx = sin(self.cart_pole.angle)
		dy = -cos(self.cart_pole.angle)
		ux, uy = dx, dy  # Pole direction
		vx, vy = -uy, ux  # Perpendicular direction
		square_points = [
			(
				pole_end_x - MASS_SQUARE_SIZE * ux - MASS_SQUARE_SIZE * vx,
				pole_end_y - MASS_SQUARE_SIZE * uy - MASS_SQUARE_SIZE * vy
			),
			(
				pole_end_x + MASS_SQUARE_SIZE * ux - MASS_SQUARE_SIZE * vx,
				pole_end_y + MASS_SQUARE_SIZE * uy - MASS_SQUARE_SIZE * vy
			),
			(
				pole_end_x + MASS_SQUARE_SIZE * ux + MASS_SQUARE_SIZE * vx,
				pole_end_y + MASS_SQUARE_SIZE * uy + MASS_SQUARE_SIZE * vy
			),
			(
				pole_end_x - MASS_SQUARE_SIZE * ux + MASS_SQUARE_SIZE * vx,
				pole_end_y - MASS_SQUARE_SIZE * uy + MASS_SQUARE_SIZE * vy
			)
		]

		pg.draw.polygon(self.scene, '#404040', square_points)

		if action == 0:
			action_str = 'Action: do nothing'
		else:
			action_str = 'Action: push left' if action == 1 else 'Action: push right'

		player_lbl = self.font.render(f'Player: {self.player}', True, 'black')
		action_lbl = self.font.render(action_str, True, 'black')
		self.scene.blit(player_lbl, (10, 10))
		self.scene.blit(action_lbl, (10, 35))

		pg.display.update()
		if terminal:
			pg.time.wait(1000)
		self.clock.tick(FPS)

	def reset(self):
		if self.training_mode:
			# If training, initialise observations with a uniformly random value
			x_init = random.uniform(-self.max_cart_x, self.max_cart_x)
			x_vel_init = random.uniform(-1, 1)
			angle_init = random.uniform(-pi, pi)
			angular_vel_init = random.uniform(-1, 1)
		else:
			x_init = x_vel_init = angular_vel_init = 0
			angle_init = pi

		self.cart_pole = CartPole(x_init, x_vel_init, angle_init, angular_vel_init)
