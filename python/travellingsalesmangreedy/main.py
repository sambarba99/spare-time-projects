"""
Greedy nearest neighbour approach to the TSP

Author: Sam Barba
Created 25/09/2022
"""

import matplotlib.pyplot as plt
import numpy as np

N_POINTS = 500

plt.rcParams['figure.figsize'] = (8, 8)

def nearest_neighbours(points):
	plot_path(order=None, all_points=points, block=False)

	# Start from central point (nearest to (50,50))
	start_idx = np.argmin(
		np.linalg.norm(np.array(points) - np.array([50, 50]), axis=1)
	)
	start = points[start_idx]
	order = [start]
	unvisited = set(points) - {start}
	while unvisited:
		nearest_neighbour = min(unvisited, key=lambda point: euclidean_dist(point, order[-1]))
		order.append(nearest_neighbour)
		unvisited.remove(nearest_neighbour)
		plot_path(order=order, all_points=points, block=False)

	plot_path(order=order, all_points=points, block=True)

def plot_path(*, order, all_points, block):
	plt.cla()
	if order:
		plt.plot(*np.array(order).T, color='red', linewidth=1, zorder=1)
	plt.scatter(*np.array(all_points).T, color='black', s=10, zorder=2)
	plt.axis('scaled')
	plt.xlabel('X')
	plt.ylabel('Y')
	if not order:
		plt.title('Start')
	else:
		total_dist = sum(euclidean_dist(point, next_point) ** 0.5
			for point, next_point in zip(order[:-1], order[1:]))
		plt.title(f'Distance: {total_dist}')
	plt.show(block=block)
	if not block:
		plt.pause(2 if not order else 1e-3)  # Pause longer at the start

def euclidean_dist(a, b):
	return np.linalg.norm(np.array(a) - np.array(b))

if __name__ == '__main__':
	points = np.random.uniform(0, 100, size=(N_POINTS, 2))
	points = list(map(tuple, points))

	nearest_neighbours(points)
