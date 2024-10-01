"""
KNN classifier

Author: Sam Barba
Created 22/10/2021
"""

from math import dist


class KNN:
	def __init__(self, k):
		self.x_train = None
		self.y_train = None
		self.k = k

	def fit(self, x, y):
		self.x_train = x
		self.y_train = y

	def predict(self, x):
		# Enumerate distances between input and all training points
		idx_and_distances = enumerate(dist(x, i) for i in self.x_train)

		# Sort in ascending order of distance
		sorted_idx_and_distances = sorted(idx_and_distances, key=lambda i: i[1])

		# Keep k nearest
		k_nearest = sorted_idx_and_distances[:self.k]

		# Get labels of nearest k samples
		k_nearest_labels = [self.y_train[idx] for idx, _ in k_nearest]

		# Prediction is the modal label
		pred = max(set(k_nearest_labels), key=k_nearest_labels.count)

		return pred
