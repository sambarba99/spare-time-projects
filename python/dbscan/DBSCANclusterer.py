# DBSCAN clusterer for dbscan.py
# Author: Sam Barba
# Created 27/12/2021

import numpy as np

UNDEFINED = -1
NOISE = 0

class DBSCANclusterer:
	def __init__(self, epsilon, minPoints):
		self.epsilon = epsilon
		self.minPoints = minPoints

	def predict(self, points):
		c = 1 # Cluster counter

		for point in points:
			if point["label"] != UNDEFINED: continue # Skip defined labels

			neighbours = [p for p in points if self.__euclideanDist(p, point) <= self.epsilon]
			if len(neighbours) < self.minPoints:
				point["label"] = NOISE
				continue

			point["label"] = c
			neighboursToExpand = [n for n in neighbours if n != point]

			for n in neighboursToExpand:
				if n["label"] == NOISE: n["label"] = c
				if n["label"] != UNDEFINED: continue

				n["label"] = c
				neighbours = [p for p in points if self.__euclideanDist(p, n) <= self.epsilon]
				if len(neighbours) >= self.minPoints:
					neighboursToExpand.extend(n for n in neighbours if n not in neighboursToExpand)

			c += 1 # Next cluster label

	def __euclideanDist(self, p1, p2):
		coords1 = np.array([p1["x"], p1["y"]])
		coords2 = np.array([p2["x"], p2["y"]])
		return ((coords1 - coords2) ** 2).sum() ** 0.5
