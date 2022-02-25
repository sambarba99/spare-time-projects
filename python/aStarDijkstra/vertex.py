# Vertex for A* and Dijkstra demo
# Author: Sam Barba
# Created 20/09/2021

class Vertex:
	def __init__(self, y, x):  # Y before X, as 2D arrays are row-major
		self.y = y
		self.x = x

		# For A* and Dijkstra
		self.is_wall = True
		self.parent_vertex = None

		# For A* only
		self.g_cost = 0
		self.h_cost = 0

		# For Dijkstra only
		self.cost = 10 ** 9  # Inf

	def get_neighbours(self, maze, maze_generation):
		dist = 2 if maze_generation else 1
		check_wall = maze_generation

		neighbours = []
		rows, cols = len(maze), len(maze[0])

		if self.x - dist >= 0 and maze[self.y][self.x - dist].is_wall == check_wall:
			neighbours.append(maze[self.y][self.x - dist])
		if self.x + dist < cols and maze[self.y][self.x + dist].is_wall == check_wall:
			neighbours.append(maze[self.y][self.x + dist])
		if self.y - dist >= 0 and maze[self.y - dist][self.x].is_wall == check_wall:
			neighbours.append(maze[self.y - dist][self.x])
		if self.y + dist < rows and maze[self.y + dist][self.x].is_wall == check_wall:
			neighbours.append(maze[self.y + dist][self.x])

		return neighbours

	def get_f_cost(self):
		return self.g_cost + self.h_cost
