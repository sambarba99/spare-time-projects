"""
Daedalus, the labyrinth creator

Author: Sam Barba
Created 20/09/2021
"""

import random

from node_maze import MazeNode

class Daedalus:
	def __init__(self, rows=99, cols=149):  # Ensure rows and cols are odd
		self.rows = rows
		self.cols = cols

	def make_maze(self):
		def remove_walls(maze, a, b):
			mid_y = (a.i + b.i) // 2
			mid_x = (a.j + b.j) // 2
			maze[mid_y][mid_x].is_wall = False
			maze[a.i][a.j].is_wall = False
			maze[b.i][b.j].is_wall = False

		graph = [[MazeNode(y, x) for x in range(self.cols)] for y in range(self.rows)]

		# Make top-left start
		graph[0][0].is_wall = False
		current = graph[0][0]
		stack = []

		while True:
			walls = current.get_neighbours(graph, maze_generation=True)

			if walls:
				next_node = random.choice(walls)
				stack.append(current)
				remove_walls(graph, current, next_node)
				current = next_node
			elif stack:
				current = stack.pop()
			else:
				return graph
