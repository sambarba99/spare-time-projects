"""
K-Means clusterer class

Author: Sam Barba
Created 21/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['figure.figsize'] = (7, 7)


class KMeans:
	def __init__(self, k):
		self.x = None
		self.k = k
		self.n_samples = 0
		self.n_features = 0
		self.clusters = None
		self.centroids = None


	def predict(self, x):
		def euclidean_dist(p1, p2):
			return np.linalg.norm(p1 - p2)


		def create_clusters():
			"""Assign the samples to the closest centroids to create clusters"""
			clusters = [[] for _ in range(self.k)]

			for sample_idx, sample in enumerate(self.x):
				distances = [euclidean_dist(sample, point) for point in self.centroids]
				centroid_idx = np.argmin(distances)
				clusters[centroid_idx].append(sample_idx)

			return clusters


		def get_centroids():
			"""Mean value of clusters"""
			centroids = np.zeros((self.k, self.n_features))

			for cluster_idx, cluster in enumerate(self.clusters):
				cluster_mean = self.x[cluster].mean(axis=0)
				centroids[cluster_idx] = cluster_mean

			return centroids


		def get_cluster_labels():
			"""For each sample, get the label of the cluster to which it was assigned"""
			labels = np.zeros(self.n_samples).astype(int)

			for cluster_idx, cluster in enumerate(self.clusters):
				labels[cluster] = cluster_idx

			return labels


		def plot(title):
			plt.cla()

			# Plot cluster points
			for idx, c in enumerate(self.clusters, start=1):
				plt.scatter(*self.x[c].T, alpha=0.5, label=f'Class {idx} ({len(x[c])} samples)')

			# Plot cluster centres
			for point in self.centroids:
				plt.scatter(*point, color='black', linewidth=3, marker='x', s=100)

			plt.axis('scaled')
			legend = plt.legend()
			for handle in legend.legend_handles:
				handle.set_alpha(1)
			plt.title(title)

			if title == 'Converged':
				plt.show()
			else:
				plt.draw()
				plt.pause(0.8)


		self.x = x
		self.n_samples = x.shape[0]
		self.n_features = x.shape[1]

		# Initially choose centroids randomly from x
		self.centroids = self.x[np.random.choice(self.n_samples, size=self.k, replace=False)]

		# Optimise clusters
		while True:
			# Create clusters by assigning samples to the closest centroids
			self.clusters = create_clusters()

			plot('Classified samples')

			# Calculate new centroids from the clusters
			centroids_prev = self.centroids
			self.centroids = get_centroids()

			# Stop if converged (no distance change since last time)
			distances = [euclidean_dist(cp, c) for cp, c in zip(centroids_prev, self.centroids)]
			if sum(distances) == 0:
				plot('Converged')
				break

			plot('Updated centroids')

		# Classify samples as the index of their clusters
		return get_cluster_labels(), self.centroids
