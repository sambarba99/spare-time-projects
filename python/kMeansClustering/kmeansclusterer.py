# K-Means clusterer for kMeansClustering.py
# Author: Sam Barba
# Created 21/11/2021

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (7, 7)

class KMeans:
	def __init__(self, k):
		self.x = None
		self.k = k
		self.num_samples = 0
		self.num_features = 0
		self.clusters = None
		self.centroids = None
		self.ax = plt.subplot()

	def predict(self, x):
		self.x = x
		self.num_samples = x.shape[0]
		self.num_features = x.shape[1]

		# Initially choose centroids randomly from x
		self.centroids = self.x[np.random.choice(self.num_samples, size=self.k, replace=False)]

		# Optimise clusters
		while True:
			# Create clusters by assigning samples to the closest centroids
			self.clusters = self.__create_clusters()

			self.__plot("Classified samples")

			# Calculate new centroids from the clusters
			centroids_prev = self.centroids
			self.centroids = self.__get_centroids()

			# Stop if converged (no distance change since last time)
			distances = [self.__euclidean_dist(cp, c) for cp, c in zip(centroids_prev, self.centroids)]
			if sum(distances) == 0:
				self.__plot("Converged")
				break

			self.__plot("Updated centroids")

		# Classify samples as the index of their clusters
		return self.__get_cluster_labels(), self.centroids

	def __create_clusters(self):
		# Assign the samples to the closest centroids to create clusters
		clusters = [[] for _ in range(self.k)]

		for sample_idx, sample in enumerate(self.x):
			distances = [self.__euclidean_dist(sample, point) for point in self.centroids]
			centroid_idx = np.argmin(distances)
			clusters[centroid_idx].append(sample_idx)

		return clusters

	def __get_centroids(self):
		# Mean value of clusters
		centroids = np.zeros((self.k, self.num_features))

		for cluster_idx, cluster in enumerate(self.clusters):
			cluster_mean = np.mean(self.x[cluster], axis=0)
			centroids[cluster_idx] = cluster_mean

		return centroids

	def __get_cluster_labels(self):
		# For each sample, get the label of the cluster to which it was assigned
		labels = np.zeros(self.num_samples).astype(int)

		for cluster_idx, cluster in enumerate(self.clusters):
			labels[cluster] = cluster_idx

		return labels

	def __plot(self, title):
		self.ax.clear()

		for idx, c in enumerate(self.clusters):
			self.ax.scatter(*self.x[c].T, alpha=0.7, label=f"Class {chr(65 + idx)}")

		for point in self.centroids:
			self.ax.scatter(*point, color="black", linewidth=3, marker="x", s=100)

		self.ax.axis("scaled")
		self.ax.legend()
		self.ax.set_title(title)
		if title != "Converged":
			plt.show(block=False)
			plt.pause(1)
		else:
			plt.show()

	def __euclidean_dist(self, p1, p2):
		# Ignore square root for faster execution
		return ((p1 - p2) ** 2).sum()
