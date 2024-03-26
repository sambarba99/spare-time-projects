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


		# Compute distance between input and all training exemplars
		distances = [euclidean_dist(x, i) for i in self.x_train]

		# Index each calculated distance, i.e.: [(0, 1.24914), (1, 0.4812)...]
		idx_and_distances = list(enumerate(distances))

		# Sort in ascending order by distance, but keep only first k training samples (nearest)
		nearest_k = sorted(idx_and_distances, key=lambda i: i[1])[:self.k]

		# Get labels of nearest k samples
		nearest_k_labels = [self.y_train[i[0]] for i in nearest_k]

		# Return mode of these
		return max(set(nearest_k_labels), key=nearest_k_labels.count)
