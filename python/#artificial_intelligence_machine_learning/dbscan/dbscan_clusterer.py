"""
DBSCAN clusterer class

Author: Sam Barba
Created 27/12/2021
"""

import numpy as np


NOISE = -1


class DBSCAN:
	def __init__(self, *, epsilon, min_samples):
		"""
		Parameters:
			epsilon: radius of local expanding clusters
			min_samples: required points within radius epsilon for it to be considered a cluster
		"""

		self.epsilon = epsilon
		self.min_samples = min_samples
		self.x = None
		self.labels = None
		self.cluster_id = 1


	def fit_predict(self, x):
		self.x = x
		self.labels = np.full(x.shape[0], NOISE)  # Initialise all points as noise

		for point_id in range(x.shape[0]):
			if self.labels[point_id] != NOISE: continue  # Skip processed points

			neighbour_ids = self.__get_neighbours(point_id)
			if len(neighbour_ids) >= self.min_samples:
				self.__expand_cluster(point_id, neighbour_ids)
				self.cluster_id += 1

		return self.labels


	def __get_neighbours(self, point_id):
		neighbour_ids = []
		for i in range(self.x.shape[0]):
			dist = np.linalg.norm(self.x[point_id] - self.x[i])
			if dist < self.epsilon:
				neighbour_ids.append(i)

		return neighbour_ids


	def __expand_cluster(self, point_id, neighbour_ids):
		self.labels[point_id] = self.cluster_id
		i = 0

		while i < len(neighbour_ids):
			neighbour_id = neighbour_ids[i]
			if self.labels[neighbour_id] == NOISE:
				self.labels[neighbour_id] = self.cluster_id
				new_neighbours = self.__get_neighbours(neighbour_id)
				if len(new_neighbours) >= self.min_samples:
					neighbour_ids.extend(new_neighbours)
			i += 1
