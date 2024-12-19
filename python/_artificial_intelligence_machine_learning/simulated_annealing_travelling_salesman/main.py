"""
Simulated Annealing applied to the TSP

Author: Sam Barba
Created 31/01/2022
"""

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['figure.figsize'] = (12, 6)
np.random.seed(1)

NUM_POINTS = 25  # (N - 1)! / 2 = 3.1x10^23 permutations
TEMP_DECREASE_FACTOR = 0.995  # Cool down by 0.5% each iteration
TEMP_THRESHOLD = 0.01  # Stop when this temperature has been reached
# NUM_POINTS = 100
# TEMP_DECREASE_FACTOR = 0.99999


def calc_distance(candidate):
	# Difference between consecutive points
	diff = np.diff(candidate, axis=0)

	# Distances between points
	distances = np.sum(diff ** 2, axis=1) ** 0.5
	loop_back_distance = sum((candidate[0] - candidate[-1]) ** 2) ** 0.5

	total_dist = sum(distances) + loop_back_distance

	return total_dist


def generate_new_candidate(candidate):
	"""
	Generates a new candidate based on an existing candidate. First, a segment is randomly chosen from
	the existing one. This segment is then either reversed or shifted with a 50/50 chance: if reversed,
	the points in the segment are reversed in order of visit. If shifted, the segment is clipped out of
	its original position and spliced in at a random point in the remainder of the path.
	"""

	start_idx, end_idx = sorted(np.random.choice(NUM_POINTS, size=2, replace=False))
	new_candidate = candidate.copy()
	segment = new_candidate[start_idx:end_idx]

	if np.random.random() < 0.5:
		# Reverse the segment
		new_candidate[start_idx:end_idx] = segment[::-1]
	else:
		# Shift the segment
		new_candidate = np.append(new_candidate[:start_idx], new_candidate[end_idx:], axis=0)
		idx = np.random.choice(len(new_candidate) + 1)
		new_candidate = np.insert(new_candidate, idx, segment, axis=0)

	return new_candidate, calc_distance(new_candidate)


def plot_progress(candidate):
	ax_dist_vs_iteration.clear()
	ax_temp_vs_iteration.clear()
	ax_current_candidate.clear()

	ax_dist_vs_iteration.plot(dist_history, linewidth=1)
	ax_dist_vs_iteration.set_xlabel('Iteration')
	ax_dist_vs_iteration.set_ylabel('Distance')
	ax_dist_vs_iteration.set_title(f'Solution distance vs iteration ({dist_history[-1]:.2f})')

	ax_temp_vs_iteration.plot(temp_history, linewidth=1)
	ax_temp_vs_iteration.set_xlabel('Iteration')
	ax_temp_vs_iteration.set_ylabel('Temperature')
	ax_temp_vs_iteration.set_title(f'Temperature vs iteration ({temp_history[-1]:.2f})')

	# Make path a loop
	coords = np.vstack((candidate, candidate[0]))

	ax_current_candidate.plot(*coords.T, color='red', linewidth=1, zorder=1)
	ax_current_candidate.scatter(*coords.T, color='black', s=18, zorder=2)
	ax_current_candidate.axis('scaled')
	ax_current_candidate.set_xlabel('x')
	ax_current_candidate.set_ylabel('y')
	ax_current_candidate.set_title(f'Current solution (iteration {iter_num} / {max_iters})')


if __name__ == '__main__':
	# Setup (initial solution is random)
	best_candidate = np.random.uniform(0, 100, size=(NUM_POINTS, 2))
	best_dist = calc_distance(best_candidate)
	temperature = NUM_POINTS  # Arbitrary start temperature
	max_iters = int(np.ceil(np.log(TEMP_THRESHOLD / temperature) / np.log(TEMP_DECREASE_FACTOR)))
	iter_num = 0
	dist_history = []
	temp_history = []

	ax_dist_vs_iteration = plt.axes([0.08, 0.57, 0.45, 0.33])
	ax_temp_vs_iteration = plt.axes([0.08, 0.1, 0.45, 0.33])
	ax_current_candidate = plt.axes([0.6, 0.1, 0.35, 0.8])

	while temperature > TEMP_THRESHOLD:
		iter_num += 1

		new_candidate, new_dist = generate_new_candidate(best_candidate)

		if new_dist < best_dist:
			best_candidate, best_dist = new_candidate, new_dist
		else:
			prob = np.exp((best_dist - new_dist) / temperature)
			if np.random.random() < prob:
				best_candidate, best_dist = new_candidate, new_dist

		temperature *= TEMP_DECREASE_FACTOR
		dist_history.append(best_dist)
		temp_history.append(temperature)

		plot_progress(new_candidate)
		plt.draw()
		plt.pause(1e-6)

	# Plot final graphs
	plot_progress(best_candidate)
	plt.show()
