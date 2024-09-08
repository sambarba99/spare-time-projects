"""
K-Means clusterer class

Author: Sam Barba
Created 21/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np


class KMeans:
	def __init__(self, k):
		self.k = k
		self.clusters = None
		self.centroids = None
		self.distance_metric = []

		_, (ax_clusters, ax_distance_metric) = plt.subplots(ncols=2, figsize=(9, 5))
		self.ax_clusters = ax_clusters
		self.ax_distance_metric = ax_distance_metric

	def fit(self, x):
		def get_cluster_labels():
			"""For each sample, get the cluster label to which it was last assigned"""

			x_cluster_labels = np.zeros(len(x), dtype=int)
			for cluster_idx, cluster in enumerate(self.clusters):
				x_cluster_labels[cluster] = cluster_idx

			return x_cluster_labels

		def get_sum_of_squares():
			"""Get the total inter-cluster sum of squares"""

			x_cluster_labels = get_cluster_labels()
			total_sum_squares = 0
			for i in range(self.k):
				cluster_points_i = x[x_cluster_labels == i]
				distances = np.linalg.norm(cluster_points_i - self.centroids[i], axis=1)
				sum_squares = (distances * distances).sum()
				total_sum_squares += sum_squares

			return total_sum_squares

		def plot(title):
			self.ax_clusters.clear()
			self.ax_distance_metric.clear()

			# Plot clusters (points and centroids)

			y_cols = get_cluster_labels()
			self.ax_clusters.scatter(*x.T, c=y_cols, cmap='jet', alpha=0.7)
			self.ax_clusters.scatter(*self.centroids.T, color='black', linewidth=3, marker='x', s=100)

			# Plot mesh and mesh point classifications

			x_min, x_max = x[:, 0].min(), x[:, 0].max()
			y_min, y_max = x[:, 1].min(), x[:, 1].max()
			xx, yy = np.meshgrid(
				np.linspace(x_min - 0.05, x_max + 0.05, 500),
				np.linspace(y_min - 0.05, y_max + 0.05, 500)
			)
			mesh_coords = np.column_stack((xx.flatten(), yy.flatten()))
			mesh_y = self.predict(mesh_coords)
			mesh_y = mesh_y.reshape(xx.shape)
			self.ax_clusters.imshow(
				mesh_y, interpolation='nearest', cmap='jet', alpha=0.2, aspect='auto', origin='lower',
				extent=(xx.min(), xx.max(), yy.min(), yy.max())
			)

			# Plot distance metric

			self.distance_metric.append(get_sum_of_squares())
			self.ax_distance_metric.plot(self.distance_metric)

			self.ax_clusters.axis('scaled')
			self.ax_clusters.set_title(title)
			self.ax_distance_metric.set_xlabel('Step no.')
			self.ax_distance_metric.set_title('Total inter-cluster sum of squares')

			if title == 'Converged':
				plt.show()
			else:
				plt.draw()
				plt.pause(0.8)


		# 1. Initially choose centroids randomly from x
		self.centroids = x[np.random.choice(len(x), size=self.k, replace=False)]

		# 2. Optimise clusters
		while True:
			# 2a. Create clusters by assigning sample indices to their nearest centroid
			self.clusters = [[] for _ in range(self.k)]
			y = self.predict(x)
			for idx, yi in enumerate(y):
				self.clusters[yi].append(idx)

			plot('Classified samples')

			# 2b. Calculate new centroids from the clusters
			centroids_prev = self.centroids.copy()

			for i in range(self.k):
				cluster_mean = x[y == i].mean(axis=0)
				self.centroids[i] = cluster_mean

			# 2c. Check for convergence (no distance change since last time)
			distances = np.linalg.norm(centroids_prev - self.centroids, axis=1)
			if sum(distances) == 0:
				plot('Converged')
				break

			plot('Updated centroids')

	def predict(self, x):
		"""Classify samples as the index of their clusters (nearest centroids)"""

		distances = np.linalg.norm(x[:, np.newaxis] - self.centroids, axis=2)
		y = distances.argmin(axis=1)
		return y
