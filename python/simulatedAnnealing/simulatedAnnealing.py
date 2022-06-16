"""
Simulated Annealing applied to 3D TSP

Author: Sam Barba
Created 31/01/2022
"""

import matplotlib.pyplot as plt
import numpy as np

NUM_TOWNS = 20  # ~ 10^18 permutations
TEMP_DECREASE_FACTOR = 0.992  # Cool down by 0.8% each iteration
TEMP_THRESHOLD = 0.1

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
		return np.linalg.norm(a - b)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def generate_new_candidate(candidate):
	"""
	Generate a new candidate based on an existing candidate. First, a segment is chosen from the
	existing candidate, then a 'coin' is flipped to choose either 'reverse' or 'shift': if reverse
	comes up, an alternative path is generated in which the towns in the chosen segment are reversed
	in order of visit. If shift, the segment is clipped out of its current position in the path and
	spliced in at a randomly chosen point in the remainder of the path.
	"""

	start, end = np.random.randint(NUM_TOWNS, size=2)
	while start == end:
		start, end = np.random.randint(NUM_TOWNS, size=2)
	start, end = min(start, end), max(start, end)

	sequence = candidate.sequence.copy()
	segment = sequence[start:end]

	if np.random.random() < 0.5:
		# Reverse the segment
		sequence[start:end] = segment[::-1]
	else:
		# Shift the segment
		sequence = np.append(sequence[:start], sequence[end:], axis=0)
		idx = np.random.choice(len(sequence) + 1)
		sequence = np.insert(sequence, idx, segment, axis=0)

	new_candidate = Path(sequence)
	return new_candidate, new_candidate.total_distance()

def plot_candidate(ax, candidate, iter_num, max_iters):
	# Make path a loop
	coords = np.vstack((candidate.sequence, candidate.sequence[0]))

	ax.clear()
	ax.plot(*coords.T, color='red', linewidth=1, zorder=1)
	ax.scatter(*coords.T, color='black', zorder=2)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title(f'Current candidate (iteration {iter_num} / {max_iters})')

def plot_dist_graph(ax, dist_history):
	ax.clear()
	ax.plot(dist_history, linewidth=1)
	ax.set_xlabel('Iteration')
	ax.set_ylabel('Distance')
	ax.set_title('Distance per iteration')

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	# Setup (initial solution is random)
	coords = np.random.uniform(0, 100, size=(NUM_TOWNS, 3))
	best_candidate = Path(coords)
	temperature = 10 * NUM_TOWNS  # Arbitrary start temperature
	best_dist = best_candidate.total_distance()
	max_iters = int(np.ceil(np.log(TEMP_THRESHOLD / temperature) / np.log(TEMP_DECREASE_FACTOR)))
	iter_num = 0
	dist_history = []

	# axes[0] = distance vs iteration
	# axes[1] = current candidate path (3D)
	_, axes = plt.subplots(nrows=2, figsize=(6, 8),
		gridspec_kw={'height_ratios': (1, 1.8), 'hspace': 0.35, 'left': 0.14, 'bottom': 0.07, 'right': 0.92, 'top': 0.92})
	axes[1] = plt.subplot(2, 1, 2, projection='3d')

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

		plot_candidate(axes[1], new_candidate, iter_num, max_iters)
		plot_dist_graph(axes[0], dist_history)
		plt.show(block=False)
		plt.pause(10 ** -6)

	# Plot best candidate and final dist graph
	plot_candidate(axes[1], best_candidate, max_iters, max_iters)
	plot_dist_graph(axes[0], dist_history)
	plt.show()
