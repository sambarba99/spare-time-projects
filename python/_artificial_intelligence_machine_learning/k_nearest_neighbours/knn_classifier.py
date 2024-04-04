"""
KNN classifier

Author: Sam Barba
Created 22/10/2021
"""

import numpy as np


class KNN:
	def __init__(self, k):
		self.x_train = None
		self.y_train = None
		self.k = k

	def fit(self, x, y):
		self.x_train = x
		self.y_train = y

	def predict(self, x):
		def euclidean_dist(a, b):
			return np.linalg.norm(np.array(a) - np.array(b))


		# Compute distance between input and all training points
		distances = [euclidean_dist(x, i) for i in self.x_train]

		# Index each calculated distance
		idx_and_distances = list(enumerate(distances))

		# Sort in ascending order by distance, and keep first k training samples (nearest)
		nearest_k = sorted(idx_and_distances, key=lambda i: i[1])[:self.k]

		# Get labels of nearest k samples
		nearest_k_labels = [self.y_train[idx] for idx, _ in nearest_k]

		# Return mode label
		return max(set(nearest_k_labels), key=nearest_k_labels.count)
