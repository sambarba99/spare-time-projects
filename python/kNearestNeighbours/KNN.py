# KNN classifier for kNearestNeighbours.py
# Author: Sam Barba
# Created 22/10/2021

import numpy as np

class KNN:
	def __init__(self, k):
		self.xTrain = None
		self.yTrain = None
		self.k = k

	def fit(self, xTrain, yTrain):
		self.xTrain = xTrain
		self.yTrain = yTrain

	def predict(self, inputs):
		# Compute distance between input and all training exemplars
		distances = [self.__euclideanDist(inputs, i) for i in self.xTrain]

		# Index each calculated distance, i.e.: [(0, 1.24914), (1, 0.4812)...]
		idxAndDistances = list(enumerate(distances))

		# Sort in ascending order by distance, but keep only first k training samples (nearest)
		nearestK = sorted(idxAndDistances, key=lambda i: i[1])[:self.k]

		# Get classifications of nearest k samples
		classesOfNearestK = [self.yTrain[i[0]] for i in nearestK]

		# Return mode of these
		return max(set(classesOfNearestK), key=classesOfNearestK.count)

	def __euclideanDist(self, x1, x2):
		x1 = np.array(x1)
		x2 = np.array(x2)
		# Ignore square root for faster execution
		return ((x1 - x2) ** 2).sum()
