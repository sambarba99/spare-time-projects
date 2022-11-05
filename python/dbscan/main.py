"""
DBSCAN demo

Author: Sam Barba
Created 27/12/2021
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, make_moons

from dbscan_clusterer import DBSCANclusterer

UNDEFINED = -1
NOISE = 0

plt.rcParams['figure.figsize'] = (8, 5)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	# 1. Load data

	choice = input('\nEnter C for circle-shaped data or M for moon-shaped data\n>>> ').upper()

	coords, _ = make_circles(n_samples=500, noise=0.08, factor=0.5, random_state=1) \
		if choice == 'C' else \
		make_moons(n_samples=500, noise=0.15, random_state=1)

	# Initialise all points with undefined labels
	points = [{'x': x, 'y': y, 'label': UNDEFINED} for x, y in coords]

	# 2. Cluster samples

	clusterer = DBSCANclusterer(epsilon=0.18, min_points=9) \
		if choice == 'C' else \
		DBSCANclusterer(epsilon=0.2, min_points=14)

	clusterer.predict(points)

	# 3. Plot clusters

	# Unique labels excluding noise
	unique_labels = sorted(list(set(p['label'] for p in points if p['label'] != NOISE)))

	# + 1 to include a 'cluster' for noise samples
	clusters = [[] for _ in range(len(unique_labels) + 1)]

	for p in points:
		# Any noise coords go into clusters[0], as noise label = 0
		coords = [p['x'], p['y']]
		clusters[p['label']].append(coords)

	for idx, c in enumerate(clusters):
		if idx == 0 and c:
			# If there's noise, plot it as black
			plt.scatter(*np.array(c).T, color='black', label=f'Noise ({len(c)} samples)')
		else:
			plt.scatter(*np.array(c).T, alpha=0.7, label=f'Class {idx} ({len(c)} samples)')

	plt.axis('scaled')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
