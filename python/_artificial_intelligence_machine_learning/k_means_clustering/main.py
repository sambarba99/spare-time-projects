"""
K-Means clustering demo

Author: Sam Barba
Created 2021-11-21
"""

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

from k_means_clusterer import KMeans


NUM_CLUSTERS = 5


if __name__ == '__main__':
	x, _ = make_blobs(n_samples=500, centers=NUM_CLUSTERS, cluster_std=2)
	scaler = MinMaxScaler()
	x = scaler.fit_transform(x)

	k_means = KMeans(NUM_CLUSTERS)
	k_means.fit(x)
	preds = k_means.predict(x)
	print(f'\nPredictions = {preds}')
