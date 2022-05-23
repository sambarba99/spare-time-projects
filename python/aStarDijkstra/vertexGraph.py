# Vertex for A* and Dijkstra demo
# Author: Sam Barba
# Created 23/05/2022

import numpy as np

class GraphVertex:
	def __init__(self, idx, x, y):
		self.idx = idx
		self.x = x
		self.y = y
		self.neighbours = None

		# For A* and Dijkstra
		self.parent_vertex = None

		# For A* only
		self.g_cost = 0
		self.h_cost = 0

		# For Dijkstra only
		self.cost = 10 ** 9  # Inf

	# For A* (Euclidean distance)
	def dist(self, other):
		a = np.array([self.x, self.y])
		b = np.array([other.x, other.y])
		return np.linalg.norm(a - b)

	# For A*
	def get_f_cost(self):
		return self.g_cost + self.h_cost
