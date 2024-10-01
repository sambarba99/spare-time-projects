"""
Node class for A* and Dijkstra demo

Author: Sam Barba
Created 20/09/2021
"""

from numpy import inf


class Node:
	def __init__(self, x, y, idx=None):
		self.x = x
		self.y = y
		self.idx = idx

		# For A* and Dijkstra
		self.is_wall = True  # For labyrinth/maze mode
		self.parent = None

		# For A* only
		self.g_cost = inf
		self.h_cost = inf

		# For Dijkstra only
		self.cost = inf

		self.neighbours = set()  # For graph (nodes/edges) mode

	@property
	def f_cost(self):
		"""For A* (F-cost = G-cost + H-cost)"""

		return self.g_cost + self.h_cost
