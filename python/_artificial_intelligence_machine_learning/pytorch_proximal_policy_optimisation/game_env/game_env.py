"""
Game environment functionality

Author: Sam Barba
Created 16/02/2023
"""

from collections import deque
from copy import deepcopy
from math import pi, sin, cos, atan2, degrees, radians

import pygame as pg

from game_env.constants import *


# Lines crossed by the car that represent progress (reward for crossing)
# Format: [x1, y1, x2, y2, active status]
REWARD_GATES = [
	[134, 383, 44, 383, 1], [134, 250, 44, 250, 0], [142, 182, 55, 140, 0],
	[175, 149, 141, 55, 0], [265, 140, 265, 45, 0], [410, 140, 410, 45, 0],
	[555, 140, 555, 45, 0], [641, 155, 699, 76, 0], [740, 226, 797, 148, 0],
	[839, 297, 895, 218, 0], [938, 369, 994, 289, 0], [1036, 441, 1093, 361, 0],
	[1136, 513, 1193, 432, 0], [1198, 567, 1288, 514, 0], [1212, 622, 1315, 622, 0],
	[1197, 669, 1280, 730, 0], [1149, 709, 1190, 806, 0], [1085, 714, 1070, 820, 0],
	[1034, 689, 961, 762, 0], [999, 642, 905, 679, 0], [991, 580, 890, 580, 0],
	[913, 405, 854, 485, 0], [806, 328, 749, 408, 0], [699, 250, 643, 331, 0],
	[598, 186, 571, 284, 0], [467, 183, 467, 278, 0], [350, 183, 350, 278, 0],
	[222, 224, 294, 296, 0], [178, 334, 281, 334, 0], [217, 430, 298, 369, 0],
	[294, 534, 374, 472, 0], [370, 637, 449, 574, 0], [385, 680, 489, 680, 0],
	[371, 711, 443, 783, 0], [283, 728, 283, 823, 0], [175, 719, 141, 814, 0],
	[142, 685, 55, 728, 0], [134, 650, 44, 650, 0], [134, 517, 44, 517, 0]
]

# These coords will be linked together to create the actual walls
INNER_WALL_VERTICES = [
	[136, 658], [136, 210], [141, 189], [153, 169], [170, 154], [187, 146],
	[204, 142], [600, 142], [617, 146], [632, 152], [1180, 549], [1197, 568],
	[1206, 588], [1210, 614], [1209, 633], [1203, 653], [1195, 669], [1180, 688],
	[1160, 702], [1140, 709], [1108, 714], [1090, 713], [1058, 704], [1034, 687],
	[1011, 663], [1000, 641], [994, 609], [993, 559], [989, 519], [976, 482],
	[960, 453], [944, 431], [920, 407], [629, 198], [608, 187], [583, 181],
	[316, 181], [294, 184], [265, 193], [231, 214], [210, 235], [191, 264],
	[179, 300], [176, 335], [181, 371], [194, 402], [214, 431], [374, 647],
	[382, 665], [383, 683], [376, 703], [360, 719], [339, 726], [206, 726],
	[187, 723], [168, 713], [152, 698], [141, 679]
]
OUTER_WALL_VERTICES = [
	[43, 685], [43, 184], [48, 152], [58, 127], [74, 101], [95, 79],
	[120, 62], [147, 50], [183, 43], [623, 43], [646, 46], [666, 52],
	[1260, 479], [1280, 499], [1300, 531], [1310, 561], [1317, 594], [1317, 629],
	[1310, 668], [1300, 697], [1285, 726], [1270, 747], [1245, 774], [1210, 799],
	[1175, 814], [1135, 823], [1105, 825], [1060, 820], [1025, 808], [994, 792],
	[960, 765], [938, 739], [916, 706], [901, 674], [891, 639], [889, 599],
	[888, 555], [886, 531], [875, 511], [849, 483], [587, 294], [568, 285],
	[546, 280], [329, 280], [307, 287], [291, 303], [284, 319], [284, 337],
	[292, 358], [468, 595], [482, 623], [490, 652], [491, 687], [483, 726],
	[466, 759], [446, 783], [417, 805], [384, 819], [352, 825], [183, 825],
	[157, 821], [128, 811], [98, 792], [78, 772], [60, 745], [48, 717]
]

vec2 = pg.math.Vector2


def find_line_intersection(line_a, line_b):
	ax1, ay1, ax2, ay2 = line_a
	bx1, by1, bx2, by2 = line_b

	denom = (ax1 - ax2) * (by1 - by2) - (ay1 - ay2) * (bx1 - bx2)

	if denom:
		t = ((ax1 - bx1) * (by1 - by2) - (ay1 - by1) * (bx1 - bx2)) / denom
		u = -((ax1 - ax2) * (ay1 - by1) - (ay1 - ay2) * (ax1 - bx1)) / denom

		if 0 <= t <= 1 and 0 <= u <= 1:
			return vec2(
				ax1 + t * (ax2 - ax1),
				ay1 + t * (ay2 - ay1)
			)

	return None


class Car:
	def __init__(self, x, y, do_rendering):
		self.do_rendering = do_rendering

		# Physical properties
		self.pos = vec2(x, y)
		self.vel = 0
		self.acc = 0
		self.direction = 0
		self.drift_vel = 0

		self.n_gates_crossed = 0
		self.drift_marks = deque(maxlen=200)

		# Car corners
		self.p1 = vec2(self.pos.x - CAR_WIDTH / 2 + 1, self.pos.y - CAR_HEIGHT / 2 + 1)
		self.p2 = vec2(self.pos.x + CAR_WIDTH / 2, self.pos.y - CAR_HEIGHT / 2 + 1)
		self.p3 = vec2(self.pos.x + CAR_WIDTH / 2, self.pos.y + CAR_HEIGHT / 2)
		self.p4 = vec2(self.pos.x - CAR_WIDTH / 2 + 1, self.pos.y + CAR_HEIGHT / 2)

		if self.do_rendering:
			self.original_img = pg.image.load('./game_env/car.png').convert()
			self.img = self.original_img  # When steering, self.img will be a rotated version of self.original_img
			self.img.set_colorkey((0, 0, 0))
			self.rect = self.img.get_rect(center=self.pos)

	def perform_action(self, action):
		"""Perform an action e.g. acceleration, and update car properties accordingly"""

		def rotate(origin, point, angle):
			ox, oy = origin
			px, py = point
			new_x = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
			new_y = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)

			return new_x, new_y


		# 1. Decode action no.

		accelerating = action in (1, 5, 6)
		decelerating = action in (2, 7, 8)
		turning_left = action in (3, 5, 7)
		turning_right = action in (4, 6, 8)

		# 2. Apply force to accelerate/decelerate if necessary

		if accelerating: self.acc = FORCE
		elif decelerating: self.acc = -FORCE
		else: self.acc = 0

		# 3. Apply steering if necessary

		if turning_left: turn_amount = -TURN_RATE * self.vel
		elif turning_right: turn_amount = TURN_RATE * self.vel
		else: turn_amount = 0

		self.direction += turn_amount
		self.direction %= (2 * pi)

		# Rotate car's image/rectangle
		if turn_amount:
			centre = (self.p1.x + self.p3.x) / 2, (self.p1.y + self.p3.y) / 2
			self.p1.update(rotate(centre, self.p1, turn_amount))
			self.p2.update(rotate(centre, self.p2, turn_amount))
			self.p3.update(rotate(centre, self.p3, turn_amount))
			self.p4.update(rotate(centre, self.p4, turn_amount))
			if self.do_rendering:
				self.img = pg.transform.rotate(self.original_img, -degrees(self.direction))

		# 4. Update velocity/drift

		self.vel += self.acc
		self.vel *= (1 - FRICTION)

		if turning_left or turning_right:
			# Function to get drift_amount from vel:
			# 	if abs(self.vel) < VEL_DRIFT_THRESHOLD, drift_amount = 0;
			# 	else, drift_amount increases linearly with a slope of DRIFT_FACTOR
			drift_amount = DRIFT_FACTOR * max(abs(self.vel), VEL_DRIFT_THRESHOLD) - VEL_DRIFT_THRESHOLD * DRIFT_FACTOR
			if turning_left:
				drift_amount *= -1
			self.drift_vel += drift_amount

		self.drift_vel *= (1 - DRIFT_FRICTION)

		if abs(self.drift_vel) > DRIFT_RENDER_THRESHOLD and self.do_rendering:
			self.drift_marks.append((self.p1.copy(), self.p2.copy(), self.p3.copy(), self.p4.copy()))

		vel_vector = vec2(
			self.vel * sin(self.direction) - self.drift_vel * cos(self.direction),
			self.vel * cos(self.direction) + self.drift_vel * sin(self.direction)
		)

		if vel_vector.length() != 0:
			vel_vector = vel_vector.normalize()

		vel_vector.x *= abs(self.vel)
		vel_vector.y *= -abs(self.vel)  # Minus because origin (0,0) is at top-left of screen

		# 5. Update position

		self.pos += vel_vector
		self.p1 += vel_vector
		self.p2 += vel_vector
		self.p3 += vel_vector
		self.p4 += vel_vector

		if self.do_rendering:
			self.rect = self.img.get_rect(center=self.pos)

	def check_crash(self, wall):
		# Lines defining the car
		line1 = *self.p1, *self.p2
		line2 = *self.p2, *self.p3
		line3 = *self.p3, *self.p4
		line4 = *self.p4, *self.p1

		return any(
			find_line_intersection(line, wall)
			for line in (line1, line2, line3, line4)
		)

	def reward_gate_crossed(self, gate):
		"""Determine if car crossed a reward gate"""

		# Line defining the car's direction
		cx1, cy1 = self.pos
		cx2 = cx1 + MAX_RAY_LENGTH * sin(self.direction)
		cy2 = cy1 - MAX_RAY_LENGTH * cos(self.direction)

		gate_line = gate[:-1]

		point = find_line_intersection((cx1, cy1, cx2, cy2), gate_line)

		return point and self.pos.distance_to(point) < CAR_HEIGHT / 2


class GameEnv:
	def __init__(self, *, do_rendering):
		self.do_rendering = do_rendering
		self.reward_gates = None

		self.walls = []
		for i in range(len(INNER_WALL_VERTICES)):
			p1 = INNER_WALL_VERTICES[i]
			p2 = INNER_WALL_VERTICES[(i + 1) % len(INNER_WALL_VERTICES)]
			self.walls.append(p1 + p2)
		for i in range(len(OUTER_WALL_VERTICES)):
			p1 = OUTER_WALL_VERTICES[i]
			p2 = OUTER_WALL_VERTICES[(i + 1) % len(OUTER_WALL_VERTICES)]
			self.walls.append(p1 + p2)

		self.car = None

		# These define the car's "vision"
		self.ray_contact_points = None
		self.ray_end_points = None  # Some rays might not intersect a wall

		if self.do_rendering:
			pg.init()
			pg.display.set_caption('Deep Reinforcement Learning self-driving car')
			self.font = pg.font.SysFont('consolas', 30)
			self.scene = pg.display.set_mode((TRACK_WIDTH, TRACK_HEIGHT))
			self.clock = pg.time.Clock()
			self.track_img = pg.image.load('./game_env/track.png').convert()

		self.reset()

	def get_state(self):
		"""Obtain the state of the car/world"""

		# Angles of rays projecting from the car
		ray_angles = [
			self.car.direction,
			self.car.direction + radians(11), self.car.direction - radians(11),
			self.car.direction + radians(45), self.car.direction - radians(45),
			self.car.direction + radians(90), self.car.direction - radians(90),
			self.car.direction + radians(135), self.car.direction - radians(135),
			self.car.direction + radians(180)
		]

		state = []
		self.ray_contact_points = []
		self.ray_end_points = []

		for ray_angle in ray_angles:
			record = MAX_RAY_LENGTH
			closest_contact_point = None
			ray_end_x = self.car.pos.x + record * sin(ray_angle)
			ray_end_y = self.car.pos.y - record * cos(ray_angle)

			for wall in self.walls:
				if pt := find_line_intersection((*self.car.pos, ray_end_x, ray_end_y), wall):
					if (dist := self.car.pos.distance_to(pt)) < record:
						record, closest_contact_point = dist, pt

			# Normalise observation: 0 is close, 1 is far
			state.append(record / MAX_RAY_LENGTH)

			# Used in rendering
			self.ray_contact_points.append(closest_contact_point)
			self.ray_end_points.append((ray_end_x, ray_end_y))

		# Agent knows its own (normalised) velocities
		# Transform from range [-max, max] to [0, 1]
		state.append((self.car.vel + MAX_GRIP_VEL) / (2 * MAX_GRIP_VEL))
		state.append((self.car.drift_vel + MAX_DRIFT_VEL) / (2 * MAX_DRIFT_VEL))

		# Find relative direction to next reward gate
		next_gate = self.reward_gates[self.car.n_gates_crossed % len(self.reward_gates)]
		next_gate_centre = vec2(
			(next_gate[0] + next_gate[2]) / 2,
			(next_gate[1] + next_gate[3]) / 2
		)
		next_gate_rel_pos = vec2(next_gate_centre.x - self.car.pos.x, next_gate_centre.y - self.car.pos.y)
		next_gate_rel_dir = degrees(self.car.direction - atan2(next_gate_rel_pos.y, next_gate_rel_pos.x)) - 90
		next_gate_rel_dir %= 360
		if next_gate_rel_dir > 180:
			next_gate_rel_dir -= 360

		state.append((next_gate_rel_dir + 180) / 360)

		return state

	def step(self, action):
		"""
		Perform an action, then return resulting info as the tuple (return, next_state, terminal)
		"""

		self.car.perform_action(action)
		next_state = self.get_state()

		# Check if car crossed the next reward gate
		gate_idx = self.car.n_gates_crossed % len(self.reward_gates)
		check_gate = self.reward_gates[gate_idx]

		# If gate active and car crossed it
		if check_gate[-1] and self.car.reward_gate_crossed(check_gate):
			# Deactivate this gate and activate next
			self.reward_gates[gate_idx][-1] = 0
			self.reward_gates[(gate_idx + 1) % len(self.reward_gates)][-1] = 1

			self.car.n_gates_crossed += 1
			return GATE_REWARD, next_state, False

		# Check if car crashed
		if any(self.car.check_crash(wall) for wall in self.walls):
			return CRASH_PENALTY, next_state, True

		# Penalise every time step
		return TIMESTEP_PENALTY, next_state, False

	def render(self, action, render_meta):
		self.scene.blit(self.track_img, self.track_img.get_rect())

		if render_meta:
			# Draw reward gates and rays

			for x1, y1, x2, y2, active in self.reward_gates:
				pg.draw.line(self.scene, (0, 255, 0) if active else (0, 0, 255), (x1, y1), (x2, y2), 2)

			# for idx, (x1, y1, x2, y2) in enumerate(self.walls):
			# 	pg.draw.line(self.scene, (255, 0, 0) if idx % 2 else (0, 0, 255), (x1, y1), (x2, y2))

			for end_p, contact_p in zip(self.ray_end_points, self.ray_contact_points):
				if contact_p:
					pg.draw.line(self.scene, (255, 255, 255), self.car.pos, contact_p)
					pg.draw.circle(self.scene, (255, 0, 0), contact_p, 4)
				else:
					pg.draw.line(self.scene, (255, 255, 255), self.car.pos, end_p)

		# Drift marks
		n_marks = len(self.car.drift_marks)
		for i in range(n_marks - 1):
			wheel_1_start = tuple(map(int, self.car.drift_marks[i][0]))
			wheel_1_end = tuple(map(int, self.car.drift_marks[i + 1][0]))

			if vec2(wheel_1_start).distance_to(vec2(wheel_1_end)) > 20:
				# Don't connect drift marks that have occurred far apart
				continue

			wheel_2_start = tuple(map(int, self.car.drift_marks[i][1]))
			wheel_2_end = tuple(map(int, self.car.drift_marks[i + 1][1]))
			wheel_3_start = tuple(map(int, self.car.drift_marks[i][2]))
			wheel_3_end = tuple(map(int, self.car.drift_marks[i + 1][2]))
			wheel_4_start = tuple(map(int, self.car.drift_marks[i][3]))
			wheel_4_end = tuple(map(int, self.car.drift_marks[i + 1][3]))
			c = int(50 * (1 - i / n_marks))
			pg.draw.line(self.scene, (c, c, c), wheel_1_start, wheel_1_end, 2)
			pg.draw.line(self.scene, (c, c, c), wheel_2_start, wheel_2_end, 2)
			pg.draw.line(self.scene, (c, c, c), wheel_3_start, wheel_3_end, 2)
			pg.draw.line(self.scene, (c, c, c), wheel_4_start, wheel_4_end, 2)

		self.scene.blit(self.car.img, self.car.rect)

		# Display WASD control (key is green if used, else grey)
		w_colour = (0, 255, 0) if action in (1, 5, 6) else (50, 50, 50)
		a_colour = (0, 255, 0) if action in (3, 5, 7) else (50, 50, 50)
		s_colour = (0, 255, 0) if action in (2, 7, 8) else (50, 50, 50)
		d_colour = (0, 255, 0) if action in (4, 6, 8) else (50, 50, 50)
		pg.draw.rect(self.scene, w_colour, pg.Rect(1092, 43, 54, 54))
		pg.draw.rect(self.scene, a_colour, pg.Rect(1035, 100, 54, 54))
		pg.draw.rect(self.scene, s_colour, pg.Rect(1092, 100, 54, 54))
		pg.draw.rect(self.scene, d_colour, pg.Rect(1149, 100, 54, 54))

		# Display laps and speed
		laps = self.car.n_gates_crossed / len(self.reward_gates)
		laps_lbl = self.font.render(f'Laps: {laps:.2f}', True, (255, 255, 255))
		speed_lbl = self.font.render(f'Speed: {self.car.vel:.1f}', True, (255, 255, 255))
		self.scene.blit(laps_lbl, dest=(1033, 179))
		self.scene.blit(speed_lbl, dest=(1033, 219))

		self.clock.tick(90)  # 90 FPS
		pg.display.update()

	def reset(self):
		self.reward_gates = deepcopy(REWARD_GATES)
		self.car = Car(89, 456, self.do_rendering)
