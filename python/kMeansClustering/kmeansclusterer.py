# K-Means clusterer for kMeansClustering.py
# Author: Sam Barba
# Created 21/11/2021

import matplotlib.pyplot as plt
import numpy as np

class KMeans:
	def __init__(self, k):
		self.x = None
		self.k = k
		self.num_samples = 0
		self.num_features = 0
		self.clusters = None
		self.centroids = None

	def predict(self, x, plot_steps=False):
		self.x = x
		self.num_samples = x.shape[0]
		self.num_features = x.shape[1]

		# Initially choose random centroids
		indices = np.random.choice(self.num_samples, self.k, replace=False)
		self.centroids = x[indices]

		# Optimise clusters
		while True:
			# Create clusters by assigning samples to the closest centroids
			self.clusters = self.__create_clusters()

			if plot_steps: self.plot("Classified samples")

			# Calculate new centroids from the clusters
			centroids_prev = self.centroids
			self.centroids = self.__get_centroids()

			# Stop if converged
			distances = [self.__euclidean_dist(cp, c) for cp, c in zip(centroids_prev, self.centroids)]
			if sum(distances) == 0:
				self.plot("Converged")
				break

			if plot_steps: self.plot("Updated centroids")

		# Classify samples as the index of their clusters
		return self.__get_cluster_labels()

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

	def plot(self, title):
		plt.figure(figsize=(8, 8))

		for c in self.clusters:
			plt.scatter(*self.x[c].T, alpha=0.7)

		for point in self.centroids:
			plt.scatter(*point, marker="x", color="black", lw=3, s=100)

		plt.legend(["Class " + str(i + 1) for i in range(len(self.clusters))])
		plt.title(title)
		if title != "Converged":
			plt.show(block=False)
			plt.pause(2)
			plt.close()
		else:
			plt.show()

	def __euclidean_dist(self, x1, x2):
		# Ignore square root for faster execution
		return ((x1 - x2) ** 2).sum()
