"""
Simulated Annealing applied to the TSP

Author: Sam Barba
Created 31/01/2022
"""

import matplotlib.pyplot as plt
import numpy as np

N_POINTS = 20  # ~ 10^18 permutations
TEMP_DECREASE_FACTOR = 0.995  # Cool down by 0.5% each iteration
TEMP_THRESHOLD = 0.1  # Stop when this temperature has been reached

plt.rcParams['figure.figsize'] = (12, 6)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def calc_distance(candidate):
	def euclidean_dist(a, b):
		return np.linalg.norm(a - b)

	sequence_zip = zip(candidate[:-1], candidate[1:])
	total_dist = sum(euclidean_dist(point, next_point) for point, next_point in sequence_zip)
	total_dist += euclidean_dist(candidate[0], candidate[-1])  # Loop back to start

	return total_dist

def generate_new_candidate(candidate):
	"""
	Generate a new candidate based on an existing candidate. First, a segment is randomly chosen from
	the existing candidate, then a 'coin' is flipped to choose either `reverse` or `shift`: if `reverse`,
	an alternative path is generated in which the points in the chosen segment are reversed in order of
	visit. If `shift`, the segment is clipped out of its original position and spliced in at a randomly
	chosen point in the remainder of the path.
	"""

	start_idx, end_idx = np.random.randint(N_POINTS, size=2)
	while start_idx == end_idx:
		end_idx = np.random.randint(N_POINTS)
	start_idx, end_idx = sorted([start_idx, end_idx])

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

def plot_dist_graph(ax, dist_history):
	ax.clear()
	ax.plot(dist_history, linewidth=1)
	ax.set_xlabel('Iteration')
	ax.set_ylabel('Distance')
	ax.set_title(f'Distance vs iteration ({dist_history[-1]:.3f})')

def plot_temp_graph(ax, temp_history):
	ax.clear()
	ax.plot(temp_history, linewidth=1)
	ax.set_xlabel('Iteration')
	ax.set_ylabel('Temperature')
	ax.set_title(f'Temperature vs iteration ({temp_history[-1]:.3f})')

def plot_candidate(ax, candidate, iter_num, max_iters):
	# Make path a loop
	coords = np.vstack((candidate, candidate[0]))

	ax.clear()
	ax.plot(*coords.T, color='red', linewidth=1, zorder=1)
	ax.scatter(*coords.T, color='black', s=18, zorder=2)
	ax.axis('scaled')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_title(f'Current candidate (iteration {iter_num} / {max_iters})')

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	np.random.seed(1)

	# Setup (initial solution is random)
	best_candidate = np.random.uniform(0, 100, size=(N_POINTS, 2))
	best_dist = calc_distance(best_candidate)
	temperature = N_POINTS  # Arbitrary start temperature
	max_iters = int(np.ceil(np.log(TEMP_THRESHOLD / temperature) / np.log(TEMP_DECREASE_FACTOR)))
	iter_num = 0
	dist_history = []
	temp_history = []

	fig = plt.figure()
	ax_dist_vs_iteration = fig.add_axes([0.08, 0.57, 0.45, 0.33])
	ax_temp_vs_iteration = fig.add_axes([0.08, 0.1, 0.45, 0.33])
	ax_current_candidate = fig.add_axes([0.6, 0.1, 0.35, 0.8])

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

		plot_dist_graph(ax_dist_vs_iteration, dist_history)
		plot_temp_graph(ax_temp_vs_iteration, temp_history)
		plot_candidate(ax_current_candidate, new_candidate, iter_num, max_iters)
		plt.show(block=False)
		plt.pause(1e-6)

	# Plot final graphs
	plot_dist_graph(ax_dist_vs_iteration, dist_history)
	plot_temp_graph(ax_temp_vs_iteration, temp_history)
	plot_candidate(ax_current_candidate, best_candidate, max_iters, max_iters)
	plt.show()
