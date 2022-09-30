"""
Fractal drawing using Lindenmayer systems (L-systems)

Author: Sam Barba
Created 19/02/2022

Controls:
Click to cycle through drawings
"""

import numpy as np
import pygame as pg
import sys

# Below are rules that generate interesting geometry
# A,B: move forward
# +: turn clockwise
# -: turn anti-clockwise
# <: decrease step size
# [: save current state
# ]: return to last saved state
# Other characters are ignored (simply placeholders)

BINARY_TREE = {'ruleset': {'axiom': 'A0', '0': '<[+A0]-A0'},
	'max-iters': 9, 'turn-angle': 45, 'start-heading': -90, 'name': 'Binary tree'}
SIERPINSKI_TRIANGLE = {'ruleset': {'axiom': 'A-B-B', 'A': 'A-B+A+B-A', 'B': 'BB'},
	'max-iters': 7, 'turn-angle': 120, 'start-heading': 0, 'name': 'Sierpinski triangle'}
SIERPINSKI_ARROWHEAD = {'ruleset': {'axiom': 'A', 'A': 'B-A-B', 'B': 'A+B+A'},
	'max-iters': 7, 'turn-angle': 60, 'start-heading': 240, 'name': 'Sierpinski arrowhead'}
KOCH_SNOWFLAKE = {'ruleset': {'axiom': 'A--A--A', 'A': 'A+A--A+A'},
	'max-iters': 5, 'turn-angle': 60, 'start-heading': 0, 'name': 'Koch snowflake'}
KOCH_ISLAND = {'ruleset': {'axiom': 'A+A+A+A', 'A': 'A-A+A+AAA-A-A+A'},
	'max-iters': 4, 'turn-angle': 90, 'start-heading': 0, 'name': 'Koch island'}
PENTAPLEXITY = {'ruleset': {'axiom': 'A++A++A++A++A', 'A': 'A++A++A+++++A-A++A'},
	'max-iters': 5, 'turn-angle': 36, 'start-heading': 36, 'name': 'Pentaplexity'}
TRIANGLES = {'ruleset': {'axiom': 'A+A+A', 'A': 'A-A+A'},
	'max-iters': 8, 'turn-angle': 120, 'start-heading': 0, 'name': 'Triangles'}
PEANO_GOSPER_CURVE = {'ruleset': {'axiom': 'A0', '0': '0+1A++1A-A0--A0A0-1A+', '1': '-A0+1A1A++1A+A0--A0-1'},
	'max-iters': 5, 'turn-angle': 60, 'start-heading': 180, 'name': 'Peano-Gosper curve'}
HILBERT_CURVE = {'ruleset': {'axiom': '0', '0': '+1A-0A0-A1+', '1': '-0A+1A1+A0-'},
	'max-iters': 8, 'turn-angle': 90, 'start-heading': 180, 'name': 'Hilbert curve'}
LEVY_C_CURVE = {'ruleset': {'axiom': 'A', 'A': '+A--A+'},
	'max-iters': 16, 'turn-angle': 45, 'start-heading': 0, 'name': 'Levy C curve'}
DRAGON_CURVE = {'ruleset': {'axiom': 'A', 'A': 'A+B', 'B': 'A-B'},
	'max-iters': 16, 'turn-angle': 90, 'start-heading': 0, 'name': 'Dragon curve'}
ASYMMETRIC_TREE_1 = {'ruleset': {'axiom': 'A', 'A': 'B+[[A]-A]-B[-BA]+A'},
	'max-iters': 6, 'turn-angle': 15, 'start-heading': -90, 'name': 'Asymmetric tree 1'}
ASYMMETRIC_TREE_2 = {'ruleset': {'axiom': 'A', 'A': 'B[+AB-[A]--A][---A]', 'B': 'BB'},
	'max-iters': 7, 'turn-angle': 22, 'start-heading': -90, 'name': 'Asymmetric tree 2'}
ASYMMETRIC_TREE_3 = {'ruleset': {'axiom': 'A', 'A': 'AA[++A][-AA]'},
	'max-iters': 7, 'turn-angle': 20, 'start-heading': -90, 'name': 'Asymmetric tree 3'}
ASYMMETRIC_TREE_4 = {'ruleset': {'axiom': 'A', 'A': 'AA+[+A-A-A]-[-A+A+A]'},
	'max-iters': 5, 'turn-angle': 20, 'start-heading': -90, 'name': 'Asymmetric tree 4'}

WIDTH, HEIGHT = 1500, 900

name_lbl_text = ''

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def generate_instructions(ruleset, n):
	"""
	Generates instructions from a ruleset applied to an initial axiom
	E.g. Dragon curve ruleset {'A': 'A+B', 'B': 'A-B'} applied to axiom 'A' 3 times:
		  A -> A+B
		A+B -> A+B+A-B
	A+B+A-B -> A+B+A-B+A+B-A-B
	"""

	instructions = ruleset['axiom']

	for _ in range(n):
		instructions = ''.join(ruleset.get(c, c) for c in instructions)

	# Remove any commands that cancel each other out
	instructions = instructions.replace('+-', '')
	instructions = instructions.replace('-+', '')

	return instructions

def execute_instructions(instructions, turn_angle, start_heading):
	# If there's no 'move forward' command, it means no drawing, so return
	if 'A' not in instructions and 'B' not in instructions:
		return False

	# State contains current x, y, heading, step size
	# Start at 0,0 with step size 1
	state = [0, 0, start_heading, 1]
	saved_states = []
	coords_to_draw = []

	scene.fill((0, 0, 0))

	# Execute the 'program', one char (command) at a time
	for cmd in instructions:
		x, y, heading, step_size = state

		match cmd:
			case cmd if cmd in 'AB':  # Move forward
				next_x = step_size * np.cos(np.radians(heading)) + x
				next_y = step_size * np.sin(np.radians(heading)) + y
				state = [next_x, next_y, heading, step_size]
				coords_to_draw.append([x, y, next_x, next_y])
			case '<': state = [x, y, heading, step_size * 0.59]  # Decrease step size
			case '+': state = [x, y, heading + turn_angle, step_size]  # Turn clockwise
			case '-': state = [x, y, heading - turn_angle, step_size]  # Turn anti-clockwise
			case '[': saved_states.append(state)  # Save current state
			case ']': state = saved_states.pop()  # Return to last saved state

	coords_to_draw = scale_and_centre_image(coords_to_draw)

	# Draw with hue increasing from red (hue 0) to yellow (60)
	for idx, (x1, y1, x2, y2) in enumerate(coords_to_draw):
		hue = idx / len(coords_to_draw) * 60
		r, g, b = hsv_to_rgb(hue, 1, 1)
		pg.draw.line(scene, (r, g, b), (x1, y1), (x2, y2))

	font = pg.font.SysFont('consolas', 18)
	name_lbl = font.render(name_lbl_text, True, (220, 220, 220))
	scene.blit(name_lbl, (10, 10))

	pg.display.update()
	return True

def scale_and_centre_image(coords):
	"""
	Calculate scale factor k: image must fill 85% of either screen's width or height,
	depending on if the image is wider than it is tall or vice-versa
	"""

	coords = np.array(coords).astype(float)
	min_x = np.min(coords[:, [0, 2]])
	max_x = np.max(coords[:, [0, 2]])
	min_y = np.min(coords[:, [1, 3]])
	max_y = np.max(coords[:, [1, 3]])

	k_x = (WIDTH * 0.85) / (max_x - min_x) if max_x > min_x else WIDTH * 0.85
	k_y = (HEIGHT * 0.85) / (max_y - min_y) if max_y > min_y else HEIGHT * 0.85
	k = min(k_x, k_y)

	coords *= k

	# Now centre image about (WIDTH / 2, HEIGHT / 2)

	mean_x = k * (min_x + max_x) / 2
	mean_y = k * (min_y + max_y) / 2

	coords[:, [0, 2]] -= mean_x - WIDTH / 2
	coords[:, [1, 3]] -= mean_y - HEIGHT / 2

	return coords

def hsv_to_rgb(h, s, v):
	"""
	HSV to RGB
	0 <= hue < 360
	0 <= saturation <= 1
	0 <= value <= 1
	"""

	c = s * v
	x = c * (1 - abs((h / 60 % 2) - 1))
	m = v - c

	r = g = b = 0
	if 0 <= h < 60: r, g, b = c, x, 0
	if 60 <= h < 120: r, g, b = x, c, 0
	if 120 <= h < 180: r, g, b = 0, c, x
	if 180 <= h < 240: r, g, b = 0, x, c
	if 240 <= h < 300: r, g, b = x, 0, c
	if 300 <= h < 360: r, g, b = c, 0, x

	return [round((col + m) * 255) for col in [r, g, b]]

def wait_for_click():
	while True:
		for event in pg.event.get():
			match event.type:
				case pg.MOUSEBUTTONDOWN: return
				case pg.QUIT: sys.exit()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	pg.init()
	pg.display.set_caption('Drawing with L-systems')
	scene = pg.display.set_mode((WIDTH, HEIGHT))

	# Draw each fractal, each from iteration 0 to its max (i.e. computer won't crash) iteration

	for fractal in [BINARY_TREE, SIERPINSKI_TRIANGLE, SIERPINSKI_ARROWHEAD, KOCH_SNOWFLAKE, KOCH_ISLAND,
		PENTAPLEXITY, TRIANGLES, PEANO_GOSPER_CURVE, HILBERT_CURVE, LEVY_C_CURVE, DRAGON_CURVE,
		ASYMMETRIC_TREE_1, ASYMMETRIC_TREE_2, ASYMMETRIC_TREE_3, ASYMMETRIC_TREE_4]:

		for i in range(fractal['max-iters'] + 1):
			name_lbl_text = f"{fractal['name']} (iteration {i}/{fractal['max-iters']})"
			instructions = generate_instructions(fractal['ruleset'], i)
			drawing_done = execute_instructions(instructions, fractal['turn-angle'], fractal['start-heading'])
			if drawing_done:
				wait_for_click()
