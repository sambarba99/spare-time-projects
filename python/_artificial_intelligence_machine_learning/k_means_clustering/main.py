"""
K-Means clustering demo

Author: Sam Barba
Created 21/11/2021
"""

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

from k_means_clusterer import KMeans


N_CLUSTERS = 5


if __name__ == '__main__':
	x, _ = make_blobs(n_samples=500, centers=N_CLUSTERS, cluster_std=2)
	scaler = MinMaxScaler()
	x = scaler.fit_transform(x)

	k_means = KMeans(N_CLUSTERS)
	k_means.fit(x)
	y_pred = k_means.predict(x)
	print(f'\ny_pred = {y_pred}')
