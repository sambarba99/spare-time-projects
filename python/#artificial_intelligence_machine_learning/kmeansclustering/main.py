"""
K-Means clustering demo

Author: Sam Barba
Created 21/11/2021
"""

import numpy as np
from sklearn.datasets import make_blobs

from k_means_clusterer import KMeans

N_CLUSTERS = 4

if __name__ == '__main__':
	x, _ = make_blobs(n_samples=500, centers=N_CLUSTERS, cluster_std=2)
	k_means = KMeans(N_CLUSTERS)
	_ = k_means.predict(x)
