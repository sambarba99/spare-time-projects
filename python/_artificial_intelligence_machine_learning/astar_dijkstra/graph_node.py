"""
Graph node for A* and Dijkstra demo

Author: Sam Barba
Created 23/05/2022
"""

import numpy as np


class GraphNode:
	def __init__(self, idx, y, x):
		self.idx = idx
		self.y = y
		self.x = x
		self.neighbours = None

		# For A* and Dijkstra
		self.parent = None

		# For A* only
		self.g_cost = 0
		self.h_cost = 0

		# For Dijkstra only
		self.cost = 1e9  # Inf

	def dist(self, other):
		"""For A* (Euclidean distance)"""

		a = np.array([self.y, self.x])
		b = np.array([other.y, other.x])
		return np.linalg.norm(a - b)

	def get_f_cost(self):
		"""For A* (F-cost = G-cost + H-cost)"""

		return self.g_cost + self.h_cost
