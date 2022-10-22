"""
K-Means clustering demo

Author: Sam Barba
Created 21/11/2021
"""

import numpy as np

from k_means_clusterer import KMeans

N_CLUSTERS = 4

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def make_random_samples(n_clusters, n_samples_per_cluster=200):
	cluster_centroid_coords = np.random.uniform(-10, 10, size=(n_clusters, 2))

	samples = []
	for cx, cy in cluster_centroid_coords:
		for _ in range(n_samples_per_cluster):
			# Generate random points radially around centroid x,y
			theta = np.random.uniform(0, 2 * np.pi)
			r = np.random.uniform(0, 5)
			sample_x = r * np.cos(theta) + cx
			sample_y = r * np.sin(theta) + cy
			samples.append([sample_x, sample_y])

	return np.array(samples)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	x = make_random_samples(N_CLUSTERS)
	k_means = KMeans(N_CLUSTERS)
	_ = k_means.predict(x)
