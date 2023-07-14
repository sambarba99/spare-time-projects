"""
Maze node for A* and Dijkstra demo

Author: Sam Barba
Created 20/09/2021
"""

class MazeNode:
	def __init__(self, i, j):
		self.i = i
		self.j = j

		# For A* and Dijkstra
		self.is_wall = True
		self.parent = None

		# For A* only
		self.g_cost = 0
		self.h_cost = 0

		# For Dijkstra only
		self.cost = 1e9  # Inf


	def get_neighbours(self, maze, *, maze_generation):
		dist = 2 if maze_generation else 1
		check_wall = maze_generation

		neighbours = []
		rows, cols = len(maze), len(maze[0])

		if self.j - dist >= 0 and maze[self.i][self.j - dist].is_wall == check_wall:
			neighbours.append(maze[self.i][self.j - dist])
		if self.j + dist < cols and maze[self.i][self.j + dist].is_wall == check_wall:
			neighbours.append(maze[self.i][self.j + dist])
		if self.i - dist >= 0 and maze[self.i - dist][self.j].is_wall == check_wall:
			neighbours.append(maze[self.i - dist][self.j])
		if self.i + dist < rows and maze[self.i + dist][self.j].is_wall == check_wall:
			neighbours.append(maze[self.i + dist][self.j])

		return neighbours


	def dist(self, other):
		"""For A* (Manhattan distance)"""

		return abs(self.j - other.j) + abs(self.i - other.i)


	def get_f_cost(self):
		"""For A* (F-cost = G-cost + H-cost)"""

		return self.g_cost + self.h_cost
