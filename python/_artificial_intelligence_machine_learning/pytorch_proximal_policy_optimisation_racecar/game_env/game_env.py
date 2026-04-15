"""
Game environment functionality

Author: Sam Barba
Created 16/02/2023
"""

from collections import deque
from copy import deepcopy
from math import pi, sin, cos, atan2, degrees, radians

import pygame as pg

from pytorch_proximal_policy_optimisation_racecar.game_env.constants import *


# Lines crossed by the car that represent progress (reward for crossing)
# Format: [x1, y1, x2, y2, active status]
REWARD_GATES = [
	[4, 640, 154, 640, 1], [4, 460, 154, 460, 0], [4, 280, 154, 280, 0], [25, 133, 159, 198, 0],
	[133, 25, 198, 159, 0], [280, 4, 280, 154, 0], [465, 4, 465, 154, 0], [650, 4, 650, 154, 0],
	[835, 4, 835, 154, 0], [1004, 55, 918, 177, 0], [1155, 163, 1067, 285, 0], [1306, 273, 1217, 393, 0],
	[1456, 381, 1368, 503, 0], [1605, 490, 1518, 611, 0], [1754, 601, 1668, 721, 0], [1904, 709, 1819, 830, 0],
	[2037, 859, 1905, 929, 0], [1931, 1043, 2081, 1043, 0], [1904, 1163, 2035, 1236, 0], [1802, 1273, 1878, 1401, 0],
	[1668, 1309, 1668, 1459, 0], [1535, 1273, 1458, 1401, 0], [1432, 1163, 1303, 1234, 0], [1255, 1043, 1405, 1043, 0],
	[1255, 882, 1405, 882, 0], [1405, 723, 1256, 752, 0], [1329, 558, 1220, 666, 0], [1176, 448, 1085, 570, 0],
	[1028, 340, 939, 464, 0], [875, 234, 809, 369, 0], [681, 211, 681, 361, 0], [521, 211, 521, 361, 0],
	[324, 238, 396, 370, 0], [237, 328, 368, 400, 0], [210, 432, 360, 432, 0], [378, 479, 258, 569, 0],
	[497, 641, 377, 732, 0], [610, 794, 490, 885, 0], [722, 945, 604, 1037, 0], [836, 1099, 716, 1190, 0],
	[734, 1236, 884, 1236, 0], [726, 1268, 857, 1340, 0], [698, 1299, 770, 1430, 0], [558, 1309, 558, 1459, 0],
	[345, 1309, 345, 1459, 0], [133, 1437, 198, 1303, 0], [25, 1329, 159, 1264, 0], [4, 1180, 154, 1180, 0],
	[4, 1000, 154, 1000, 0], [4, 820, 154, 820, 0]
]

# These coords will be linked together to create the actual walls
INNER_WALL_VERTICES = [
	[155, 1240], [155, 222], [165, 193], [181, 173], [203, 160], [226, 155], [860, 155], [886, 161], [904, 171],
	[1835, 844], [1856, 864], [1879, 890], [1902, 928], [1914, 957], [1921, 980], [1928, 1015], [1930, 1043],
	[1928, 1073], [1921, 1111], [1907, 1151], [1893, 1179], [1875, 1204], [1854, 1228], [1816, 1260], [1768, 1287],
	[1725, 1300], [1683, 1306], [1652, 1306], [1614, 1300], [1564, 1285], [1521, 1261], [1489, 1234], [1459, 1201],
	[1433, 1158], [1417, 1118], [1411, 1089], [1408, 1053], [1408, 705], [1402, 671], [1389, 630], [1369, 595],
	[1337, 559], [908, 249], [873, 229], [837, 216], [797, 209], [421, 209], [385, 214], [339, 229], [299, 253],
	[266, 283], [238, 322], [221, 358], [211, 397], [209, 418], [209, 449], [215, 488], [227, 524], [246, 557],
	[723, 1204], [731, 1226], [731, 1249], [722, 1273], [707, 1290], [693, 1300], [670, 1306], [226, 1306], [203, 1302],
	[181, 1288], [165, 1269]
]
OUTER_WALL_VERTICES = [
	[2, 1250], [2, 210], [11, 161], [23, 128], [51, 86], [82, 54], [115, 31], [156, 13], [210, 2], [886, 2], [918, 8],
	[954, 20], [988, 38], [1921, 715], [1964, 753], [1999, 794], [2027, 834], [2054, 890], [2070, 936], [2079, 981],
	[2083, 1025], [2083, 1062], [2079, 1107], [2070, 1152], [2053, 1203], [2024, 1260], [1999, 1296], [1962, 1339],
	[1923, 1373], [1883, 1401], [1841, 1423], [1791, 1442], [1747, 1453], [1706, 1459], [1679, 1460], [1658, 1460],
	[1626, 1459], [1567, 1448], [1523, 1434], [1485, 1418], [1433, 1388], [1381, 1346], [1335, 1293], [1303, 1242],
	[1281, 1197], [1268, 1158], [1258, 1112], [1253, 1063], [1253, 720], [1247, 697], [1235, 681], [812, 375],
	[796, 366], [775, 363], [425, 363], [398, 372], [380, 387], [368, 405], [363, 423], [363, 442], [369, 461],
	[853, 1119], [871, 1155], [881, 1187], [885, 1218], [885, 1255], [877, 1299], [864, 1335], [839, 1374], [811, 1404],
	[778, 1429], [745, 1445], [704, 1457], [674, 1460], [210, 1460], [153, 1448], [114, 1431], [82, 1408], [51, 1376],
	[23, 1331], [11, 1299]
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
	def __init__(self, do_rendering):
		self.do_rendering = do_rendering

		# Physical properties
		self.pos = vec2(START_X, START_Y)
		self.vel = 0
		self.acc = 0
		self.heading = 0  # Heading = direction pointed by car
		self.drift_vel = 0

		self.num_gates_crossed = 0
		self.drift_marks = deque(maxlen=200)

		# Car corners
		self.p1 = vec2(self.pos.x - CAR_WIDTH // 2 + 2, self.pos.y - CAR_HEIGHT // 2 + 1)
		self.p2 = vec2(self.pos.x + CAR_WIDTH // 2 - 2, self.pos.y - CAR_HEIGHT // 2 + 1)
		self.p3 = vec2(self.pos.x + CAR_WIDTH // 2 - 2, self.pos.y + CAR_HEIGHT // 2 - 1)
		self.p4 = vec2(self.pos.x - CAR_WIDTH // 2 + 2, self.pos.y + CAR_HEIGHT // 2 - 1)

		if self.do_rendering:
			self.original_img = pg.image.load('./game_env/car.png').convert_alpha()
			self.img = self.original_img  # When steering, self.img will be a rotated version of self.original_img

	def perform_action(self, action):
		"""Perform an action e.g. acceleration, and update car properties accordingly"""

		def rotate(origin, point, angle):
			ox, oy = origin
			px, py = point
			new_x = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
			new_y = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)

			return new_x, new_y


		# Decode action num

		accelerating = action in (1, 5, 6)
		decelerating = action in (2, 7, 8)
		turning_left = action in (3, 5, 7)
		turning_right = action in (4, 6, 8)

		# Apply force to accelerate/decelerate if necessary

		if accelerating:
			self.acc = FORCE
		elif decelerating:
			self.acc = -FORCE
		else:
			self.acc = 0

		# Apply steering if necessary

		if turning_left:
			turn_amount = -TURN_RATE * self.vel
		elif turning_right:
			turn_amount = TURN_RATE * self.vel
		else:
			turn_amount = 0

		self.heading += turn_amount
		self.heading %= (2 * pi)

		# Rotate car's vertices and image
		if turn_amount:
			centre = (self.p1.x + self.p3.x) / 2, (self.p1.y + self.p3.y) / 2
			self.p1.update(rotate(centre, self.p1, turn_amount))
			self.p2.update(rotate(centre, self.p2, turn_amount))
			self.p3.update(rotate(centre, self.p3, turn_amount))
			self.p4.update(rotate(centre, self.p4, turn_amount))
			if self.do_rendering:
				self.img = pg.transform.rotate(self.original_img, -degrees(self.heading))

		# Update velocity/drift

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
			self.vel * sin(self.heading) - self.drift_vel * cos(self.heading),
			self.vel * cos(self.heading) + self.drift_vel * sin(self.heading)
		)

		if vel_vector.magnitude() != 0:
			vel_vector = vel_vector.normalize()

		vel_vector.x *= abs(self.vel)
		vel_vector.y *= -abs(self.vel)  # Minus because origin (0,0) is at top-left of screen

		# Update position

		self.pos += vel_vector
		self.p1 += vel_vector
		self.p2 += vel_vector
		self.p3 += vel_vector
		self.p4 += vel_vector

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
		cx2 = cx1 + MAX_RAY_LENGTH * sin(self.heading)
		cy2 = cy1 - MAX_RAY_LENGTH * cos(self.heading)

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
		self.camera_pos = None
		self.screen_centre = vec2(SCENE_WIDTH / 2, SCENE_HEIGHT / 2)

		# These define the car's "vision" and are only used when rendering
		self.ray_contact_points = None
		self.ray_end_points = None  # Some rays might not intersect a wall

		if self.do_rendering:
			pg.init()
			pg.display.set_caption('Deep Reinforcement Learning self-driving car')
			self.font = pg.font.SysFont('consolas', 26)
			self.scene = pg.display.set_mode((SCENE_WIDTH, SCENE_HEIGHT))
			self.clock = pg.time.Clock()
			self.track_img = pg.image.load('./game_env/track.png').convert()

		self.reset()

	def get_state(self):
		"""Obtain the state of the car/world"""

		# Angles of rays projecting from the car
		ray_angles = [
			self.car.heading,
			self.car.heading + radians(11), self.car.heading - radians(11),
			self.car.heading + radians(45), self.car.heading - radians(45),
			self.car.heading + radians(90), self.car.heading - radians(90),
			self.car.heading + radians(135), self.car.heading - radians(135),
			self.car.heading + radians(180)
		]

		state = []
		self.ray_contact_points = []
		self.ray_end_points = []

		for ray_angle in ray_angles:
			record = MAX_RAY_LENGTH
			nearest_contact_point = None
			ray_end_x = self.car.pos.x + record * sin(ray_angle)
			ray_end_y = self.car.pos.y - record * cos(ray_angle)

			for wall in self.walls:
				if pt := find_line_intersection((*self.car.pos, ray_end_x, ray_end_y), wall):
					if (dist := self.car.pos.distance_to(pt)) < record:
						record, nearest_contact_point = dist, pt

			# Normalise observation: 0 is close, 1 is far
			state.append(record / MAX_RAY_LENGTH)

			# Used in rendering
			self.ray_contact_points.append(nearest_contact_point)
			self.ray_end_points.append((ray_end_x, ray_end_y))

		# Agent knows its own velocities (normalised)
		# Transform from range [-max, max] to [0, 1]
		state.append((self.car.vel + MAX_GRIP_VEL) / (2 * MAX_GRIP_VEL))
		state.append((self.car.drift_vel + MAX_DRIFT_VEL) / (2 * MAX_DRIFT_VEL))

		# Find relative direction to next reward gate
		next_gate = self.reward_gates[self.car.num_gates_crossed % len(self.reward_gates)]
		next_gate_centre = vec2(
			(next_gate[0] + next_gate[2]) / 2,
			(next_gate[1] + next_gate[3]) / 2
		)
		next_gate_rel_pos = vec2(next_gate_centre.x - self.car.pos.x, next_gate_centre.y - self.car.pos.y)
		next_gate_rel_dir = degrees(self.car.heading - atan2(next_gate_rel_pos.y, next_gate_rel_pos.x)) - 90
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
		gate_idx = self.car.num_gates_crossed % len(self.reward_gates)
		check_gate = self.reward_gates[gate_idx]

		# If gate active and car crossed it
		if check_gate[-1] and self.car.reward_gate_crossed(check_gate):
			# Deactivate this gate and activate next
			self.reward_gates[gate_idx][-1] = 0
			self.reward_gates[(gate_idx + 1) % len(self.reward_gates)][-1] = 1

			self.car.num_gates_crossed += 1
			return GATE_REWARD, next_state, False

		# Check if car crashed
		if any(self.car.check_crash(wall) for wall in self.walls):
			return CRASH_PENALTY, next_state, True

		# Penalise every time step
		return TIMESTEP_PENALTY, next_state, False

	def render(self, action, render_meta, terminal):
		self.scene.fill('black')

		# Compute lookahead position based on car vel and heading
		lookahead_distance = LOOKAHEAD_FACTOR * abs(self.car.vel)  # E.g. For max vel (~11), look ahead ~132px
		forward = vec2(sin(self.car.heading), -cos(self.car.heading))
		lookahead_pos = self.car.pos + forward * lookahead_distance

		# Compute camera follow speed based on car vel
		speed_ratio = min(abs(self.car.vel) / MAX_GRIP_VEL, 1)
		camera_follow_speed = MIN_FOLLOW_SPEED + (MAX_FOLLOW_SPEED - MIN_FOLLOW_SPEED) * speed_ratio

		# Compute camera position based on lookahead pos and camera follow speed
		self.camera_pos += (lookahead_pos - self.camera_pos) * camera_follow_speed

		camera_offset = self.screen_centre - self.camera_pos
		car_screen_pos = self.car.pos + camera_offset

		track_rect = self.track_img.get_rect()
		track_rect.topleft = camera_offset  # Shift track
		self.scene.blit(self.track_img, track_rect)

		if render_meta:
			# Draw reward gates, rays (+ walls, car boundary)

			for x1, y1, x2, y2, active in self.reward_gates:
				start = vec2(x1, y1) + camera_offset
				end = vec2(x2, y2) + camera_offset
				pg.draw.line(self.scene, 'green' if active else 'blue', start, end, 2)
				# lbl1 = self.font.render(f'{x1}, {y1}', True, 'white')
				# lbl2 = self.font.render(f'{x2}, {y2}', True, 'white')
				# self.scene.blit(lbl1, (x1 + camera_offset.x - 50, y1 + camera_offset.y))
				# self.scene.blit(lbl2, (x2 + camera_offset.x - 50, y2 + camera_offset.y))

			# for idx, (x1, y1, x2, y2) in enumerate(self.walls):
			# 	start = vec2(x1, y1) + camera_offset
			# 	end = vec2(x2, y2) + camera_offset
			# 	pg.draw.line(self.scene, 'red' if idx % 2 else 'blue', start, end)

			# pg.draw.line(self.scene, 'red', self.car.p1 + camera_offset, self.car.p2 + camera_offset)
			# pg.draw.line(self.scene, 'orange', self.car.p2 + camera_offset, self.car.p3 + camera_offset)
			# pg.draw.line(self.scene, 'yellow', self.car.p3 + camera_offset, self.car.p4 + camera_offset)
			# pg.draw.line(self.scene, 'green', self.car.p4 + camera_offset, self.car.p1 + camera_offset)

			for end_p, contact_p in zip(self.ray_end_points, self.ray_contact_points):
				if contact_p:
					pg.draw.line(self.scene, 'white', car_screen_pos, contact_p + camera_offset)
					pg.draw.circle(self.scene, 'red', contact_p + camera_offset, 4)
				else:
					pg.draw.line(self.scene, 'white', car_screen_pos, end_p + camera_offset)

		# Drift marks
		num_marks = len(self.car.drift_marks)
		for i in range(num_marks - 1):
			wheel_1_start = tuple(map(int, self.car.drift_marks[i][0] + camera_offset))
			wheel_1_end = tuple(map(int, self.car.drift_marks[i + 1][0] + camera_offset))

			if vec2(wheel_1_start).distance_to(vec2(wheel_1_end)) > 20:
				# Don't connect drift marks that have occurred far apart
				continue

			wheel_2_start = tuple(map(int, self.car.drift_marks[i][1] + camera_offset))
			wheel_2_end = tuple(map(int, self.car.drift_marks[i + 1][1] + camera_offset))
			wheel_3_start = tuple(map(int, self.car.drift_marks[i][2] + camera_offset))
			wheel_3_end = tuple(map(int, self.car.drift_marks[i + 1][2] + camera_offset))
			wheel_4_start = tuple(map(int, self.car.drift_marks[i][3] + camera_offset))
			wheel_4_end = tuple(map(int, self.car.drift_marks[i + 1][3] + camera_offset))
			c = int(50 * (1 - i / num_marks))
			pg.draw.line(self.scene, (c, c, c), wheel_1_start, wheel_1_end, 2)
			pg.draw.line(self.scene, (c, c, c), wheel_2_start, wheel_2_end, 2)
			pg.draw.line(self.scene, (c, c, c), wheel_3_start, wheel_3_end, 2)
			pg.draw.line(self.scene, (c, c, c), wheel_4_start, wheel_4_end, 2)

		car_rect = self.car.img.get_rect(center=car_screen_pos)
		self.scene.blit(self.car.img, car_rect)

		# Display WASD control (key is green if used, else grey)
		if terminal:
			w_colour = a_colour = s_colour = d_colour = 'red'
		else:
			w_colour = 'green' if action in (1, 5, 6) else (30, 30, 30)
			a_colour = 'green' if action in (3, 5, 7) else (30, 30, 30)
			s_colour = 'green' if action in (2, 7, 8) else (30, 30, 30)
			d_colour = 'green' if action in (4, 6, 8) else (30, 30, 30)
		pg.draw.rect(self.scene, w_colour, pg.Rect(103, 50, 50, 50))
		pg.draw.rect(self.scene, a_colour, pg.Rect(50, 103, 50, 50))
		pg.draw.rect(self.scene, s_colour, pg.Rect(103, 103, 50, 50))
		pg.draw.rect(self.scene, d_colour, pg.Rect(156, 103, 50, 50))

		# Display laps and speed
		laps = self.car.num_gates_crossed / len(self.reward_gates)
		laps_lbl = self.font.render(f'Laps: {laps:.2f}', True, 'white')
		speed_lbl = self.font.render(f'Speed: {self.car.vel:.1f}', True, 'white')
		self.scene.blit(laps_lbl, (49, 171))
		self.scene.blit(speed_lbl, (49, 209))

		pg.display.update()
		if terminal:
			pg.time.wait(800)
		self.clock.tick(FPS)

	def reset(self):
		self.reward_gates = deepcopy(REWARD_GATES)
		self.car = Car(self.do_rendering)
		self.camera_pos = vec2(self.car.pos)
