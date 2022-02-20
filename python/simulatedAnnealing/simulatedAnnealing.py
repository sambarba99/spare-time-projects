# Simulated Annealing applied to TSP
# Author: Sam Barba
# Created 31/01/2022

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import sys

NUM_TOWNS = 30 # ~ 10^32 permutations
TEMP_DECREASE_FACTOR = 0.999 # Cool down by 0.1% each iteration
TEMP_THRESHOLD = 0.1
USE_POLYGON_PATTERN = False

best_candidate = temperature = None

# ---------------------------------------------------------------------------------------------------- #
# ---------------------------------------------  CLASSES  -------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

class Path:
	def __init__(self, sequence):
		self.sequence = sequence

	def total_distance(self):
		total = 0
		for i in range(NUM_TOWNS - 1):
			town1, town2 = self.sequence[i:i + 2]
			total += self.__euclidean_dist(town1, town2)
		total += self.__euclidean_dist(self.sequence[0], self.sequence[-1])

		return total

	def __euclidean_dist(self, a, b):
		return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def initialise():
	global best_candidate, temperature

	# Y coords are offset slightly so as to not obscure pygame labels
	if USE_POLYGON_PATTERN:
		# Generate towns in the shape of a regular polygon
		angles = np.linspace(0, 2 * np.pi, NUM_TOWNS)
		x = [250 * np.cos(a) + 300 for a in angles]
		y = [250 * np.sin(a) + 320 for a in angles]
	else:
		# Generate random town coords
		x = np.random.randint(20, 580, size=NUM_TOWNS)
		y = np.random.randint(70, 580, size=NUM_TOWNS)

	coords = list(zip(x, y))
	np.random.shuffle(coords)
	best_candidate = Path(coords) # Initial solution (random)
	temperature = 10 * NUM_TOWNS # Arbitrary start temperature

# Generate a new candidate based on an existing candidate. First, a segment is chosen from the existing
# candidate, then a 'coin' is flipped to choose either 'reverse' or 'shift': If reverse comes up, an
# alternative path is generated in which the towns in the chosen segment are reversed in order of visit.
# If shift, the segment is clipped out of its current position in the path and spliced in at a randomly
# chosen point in the remainder of the path.
def generate_new_candidate(candidate):
	start, end = np.random.randint(0, NUM_TOWNS, size=2)
	while start == end:
		start, end = np.random.randint(0, NUM_TOWNS, size=2)
	start, end = min(start, end), max(start, end)

	sequence = candidate.sequence[:]
	segment = sequence[start:end]

	if np.random.uniform() < 0.5: # Reverse
		sequence[start:end] = segment[::-1]
	else: # Shift
		sequence = sequence[:start] + sequence[end:]
		idx = np.random.randint(len(sequence) + 1)
		sequence[idx:idx] = segment

	new_candidate = Path(sequence)
	return new_candidate, new_candidate.total_distance()

def draw_solution(candidate, total_distance, iteration, max_iters):
	scene.fill((20, 20, 20))

	# Draw connecting lines, then dots (towns) on top
	for i in range(NUM_TOWNS - 1):
		town1, town2 = candidate.sequence[i:i + 2]
		pg.draw.line(scene, (220, 220, 220), town1, town2)
	first, last = candidate.sequence[0], candidate.sequence[-1]
	pg.draw.line(scene, (220, 220, 220), first, last)

	for town in candidate.sequence:
		pg.draw.circle(scene, (230, 20, 20), town, 5)

	# Draw labels for iteration number, temperature, current candidate total distance
	font = pg.font.SysFont("consolas", 16)
	iteration_lbl_text = f"  Iteration: {iteration}/{max_iters}"
	temp_lbl_text = f"Temperature: {temperature:.5f}"
	distance_lbl_text = f"   Distance: {total_distance:.2f}"
	iteration_lbl = font.render(iteration_lbl_text, True, (220, 220, 220))
	temp_lbl = font.render(temp_lbl_text, True, (220, 220, 220))
	distance_lbl = font.render(distance_lbl_text, True, (220, 220, 220))
	scene.blit(iteration_lbl, (5, 5))
	scene.blit(temp_lbl, (5, 25))
	scene.blit(distance_lbl, (5, 45))

	pg.display.update()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

pg.init()
pg.display.set_caption("Simulated Annealing for TSP")
scene = pg.display.set_mode((600, 600))

# Setup
initialise()
best_dist = best_candidate.total_distance()
best_distance_history = []
iter_num = 0
max_iters = int(np.ceil(np.log(TEMP_THRESHOLD / temperature) / np.log(TEMP_DECREASE_FACTOR)))

while temperature > TEMP_THRESHOLD:
	for event in pg.event.get(): # Handle pygame events
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)

	iter_num += 1

	new_candidate, new_dist = generate_new_candidate(best_candidate)

	draw_solution(new_candidate, new_dist, iter_num, max_iters)

	if new_dist < best_dist:
		best_candidate, best_dist = new_candidate, new_dist
	else:
		prob = np.exp((best_dist - new_dist) / temperature)
		if np.random.uniform() < prob:
			best_candidate, best_dist = new_candidate, new_dist

	temperature *= TEMP_DECREASE_FACTOR
	best_distance_history.append(best_dist)

# Plot best candidate and evolution graph

draw_solution(best_candidate, best_dist, max_iters, max_iters)

plt.figure(figsize=(8, 6))
plt.plot(range(len(best_distance_history)), best_distance_history, linewidth=1)
plt.xlabel("Iteration")
plt.ylabel("Best distance")
plt.title("Best distance over time")
plt.show()
