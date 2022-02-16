# DBSCAN clusterer for dbscan.py
# Author: Sam Barba
# Created 27/12/2021

import numpy as np

UNDEFINED = -1
NOISE = 0

class DBSCANclusterer:
	def __init__(self, epsilon, min_points):
		self.epsilon = epsilon
		self.minPoints = min_points

	def predict(self, points):
		c = 1 # Cluster counter

		for point in points:
			if point["label"] != UNDEFINED: continue # Skip defined labels

			neighbours = [p for p in points if self.__euclidean_dist(p, point) <= self.epsilon]
			if len(neighbours) < self.minPoints:
				point["label"] = NOISE
				continue

			point["label"] = c
			neighbours_to_expand = [n for n in neighbours if n != point]

			for n in neighbours_to_expand:
				if n["label"] == NOISE: n["label"] = c
				if n["label"] != UNDEFINED: continue

				n["label"] = c
				neighbours = [p for p in points if self.__euclidean_dist(p, n) <= self.epsilon]
				if len(neighbours) >= self.minPoints:
					neighbours_to_expand.extend(n for n in neighbours if n not in neighbours_to_expand)

			c += 1 # Next cluster label

	def __euclidean_dist(self, p1, p2):
		coords1 = np.array([p1["x"], p1["y"]])
		coords2 = np.array([p2["x"], p2["y"]])
		return ((coords1 - coords2) ** 2).sum() ** 0.5
