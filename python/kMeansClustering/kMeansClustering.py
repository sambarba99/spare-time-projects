# K-Means clustering demo
# Author: Sam Barba
# Created 21/11/2021

from kmeansclusterer import KMeans
import numpy as np

NUM_CLUSTERS = 4

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def make_random_samples(num_clusters, num_samples_per_cluster=300):
	cluster_centroid_coords = np.random.uniform(-100, 100, size=(num_clusters, 2))

	samples = []
	for c in cluster_centroid_coords:
		for _ in range(num_samples_per_cluster):
			# Generate random points radially around centroid x,y
			theta = np.random.uniform(0, 2 * np.pi)
			r = np.random.uniform(0, 50)
			x = r * np.cos(theta) + c[0]
			y = r * np.sin(theta) + c[1]
			samples.append([x, y])

	return np.array(samples)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

x = make_random_samples(NUM_CLUSTERS)

k_means = KMeans(NUM_CLUSTERS)
y_pred = k_means.predict(x, True)
