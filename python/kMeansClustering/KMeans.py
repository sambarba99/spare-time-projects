# K-Means model for kMeansClustering.py
# Author: Sam Barba
# Created 21/11/2021

import matplotlib.pyplot as plt
import numpy as np

class KMeans:
	def __init__(self, k):
		self.x = None
		self.k = k
		self.numSamples = 0
		self.numFeatures = 0
		self.clusters = None
		self.centroids = None

	def predict(self, x, plotSteps=False):
		self.x = x
		self.numSamples = x.shape[0]
		self.numFeatures = x.shape[1]

		# Initially choose random centroids
		indices = np.random.choice(self.numSamples, self.k, replace=False)
		self.centroids = x[indices]

		# Optimise clusters
		while True:
			# Create clusters by assigning samples to the closest centroids
			self.clusters = self.__createClusters()

			if plotSteps: self.plot("Classified samples")

			# Calculate new centroids from the clusters
			centroidsPrev = self.centroids
			self.centroids = self.__getCentroids()

			# Stop if converged
			distances = [self.__euclideanDistance(cp, c) for cp, c in zip(centroidsPrev, self.centroids)]
			if sum(distances) == 0:
				self.plot("Converged")
				break

			if plotSteps: self.plot("Updated centroids")

		# Classify samples as the index of their clusters
		return self.__getClusterLabels()

	def __createClusters(self):
		# Assign the samples to the closest centroids to create clusters
		clusters = [[] for _ in range(self.k)]

		for sampleIdx, sample in enumerate(self.x):
			distances = [self.__euclideanDistance(sample, point) for point in self.centroids]
			centroidIdx = np.argmin(distances)
			clusters[centroidIdx].append(sampleIdx)

		return clusters

	def __getCentroids(self):
		# Mean value of clusters
		centroids = np.zeros((self.k, self.numFeatures))

		for clusterIdx, cluster in enumerate(self.clusters):
			clusterMean = np.mean(self.x[cluster], axis=0)
			centroids[clusterIdx] = clusterMean

		return centroids

	def __getClusterLabels(self):
		# For each sample, get the label of the cluster to which it was assigned
		labels = np.zeros(self.numSamples).astype(int)

		for clusterIdx, cluster in enumerate(self.clusters):
			labels[cluster] = clusterIdx

		return labels

	def plot(self, title):
		plt.figure(figsize=(8, 8))

		for c in self.clusters:
			plt.scatter(*self.x[c].T, alpha=0.7)

		for point in self.centroids:
			plt.scatter(*point, marker="x", color="black", lw=3, s=100)

		plt.title(title)
		if title != "Converged":
			plt.show(block=False)
			plt.pause(2)
			plt.close()
		else:
			plt.show()

	def __euclideanDistance(self, x1, x2):
		# Ignore square root for faster execution
		return ((x1 - x2) ** 2).sum()
