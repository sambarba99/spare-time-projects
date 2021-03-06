"""
Daedalus, the labyrinth creator

Author: Sam Barba
Created 20/09/2021
"""

import random
from vertexMaze import MazeVertex

class Daedalus:
	def __init__(self, rows=59, cols=99):  # Ensure rows and cols are odd
		self.rows = rows
		self.cols = cols

	def make_maze(self):
		graph = [[MazeVertex(y, x) for x in range(self.cols)] for y in range(self.rows)]

		# Make top-left start
		graph[0][0].is_wall = False
		current = graph[0][0]
		stack = []

		while True:
			walls = current.get_neighbours(graph, maze_generation=True)

			if walls:
				next_v = random.choice(walls)
				stack.append(current)
				self.__remove_walls(graph, current, next_v)
				current = next_v
			elif stack:
				current = stack.pop()
			else:
				return graph

	def __remove_walls(self, maze, a, b):
		mid_y = (a.y + b.y) // 2
		mid_x = (a.x + b.x) // 2
		maze[mid_y][mid_x].is_wall = False
		maze[a.y][a.x].is_wall = False
		maze[b.y][b.x].is_wall = False
