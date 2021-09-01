"""
Demonstration of the DBSCAN algorithm

Author: Sam Barba
Created 27/12/2021
"""

from dbscan_clusterer import DBSCANclusterer
import matplotlib.pyplot as plt
import numpy as np

UNDEFINED = -1
NOISE = 0

plt.rcParams['figure.figsize'] = (8, 5)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	# 1. Load file coords

	coords = np.genfromtxt('C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\dbscanData.txt',
		dtype=str, delimiter='\n')
	# Skip header and convert to floats
	coords = [row.split() for row in coords[1:]]
	coords = np.array(coords).astype(float)

	# Initialise all points with undefined labels
	points = [{'x': x, 'y': y, 'label': UNDEFINED} for x, y in coords]

	# 2. Cluster samples

	clusterer = DBSCANclusterer(epsilon=10, min_points=12)
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
		if idx == 0 and len(c) > 0:
			# If there's noise, plot it as black
			plt.scatter(*np.array(c).T, color='black', label=f'Noise ({len(c)} samples)')
		else:
			plt.scatter(*np.array(c).T, alpha=0.7, label=f'Class {idx} ({len(c)} samples)')

	plt.axis('scaled')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
