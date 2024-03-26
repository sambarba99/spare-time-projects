"""
DBSCAN demo

Author: Sam Barba
Created 27/12/2021
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons

from dbscan_clusterer import DBSCAN


plt.rcParams['figure.figsize'] = (8, 5)


if __name__ == '__main__':
	# 1. Load data

	choice = input('\nEnter C for circle-shaped data or M for moon-shaped data\n>>> ').upper()

	x, _ = make_circles(n_samples=500, noise=0.1, factor=0.4, random_state=1) \
		if choice == 'C' else \
		make_moons(n_samples=500, noise=0.1, random_state=1)

	# 2. Cluster samples

	clusterer = DBSCAN(epsilon=0.2, min_samples=9) \
		if choice == 'C' else \
		DBSCAN(epsilon=0.14, min_samples=5)

	labels = clusterer.fit_predict(x)

	# 3. Plot clusters and noise

	unique_labels = sorted(set(labels))

	for label in unique_labels:
		member_mask = labels == label
		member_points = x[member_mask]
		count = len(member_points)

		if label == -1:  # Noise
			plt.scatter(*member_points.T, label=f'Noise ({count} points)', color='black', marker='x')
		else:
			plt.scatter(*member_points.T, label=f'Cluster {label} ({count} points)', alpha=0.5)

	legend = plt.legend()
	for handle in legend.legend_handles:
		handle.set_alpha(1)
	plt.axis('scaled')
	plt.show()
