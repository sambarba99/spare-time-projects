"""
Daedalus, the labyrinth creator

Author: Sam Barba
Created 20/09/2021
"""

import random

from maze_node import MazeNode


def make_maze(rows, cols):
	def remove_walls(maze, a, b):
		mid_y = (a.i + b.i) // 2
		mid_x = (a.j + b.j) // 2
		maze[mid_y][mid_x].is_wall = False
		maze[a.i][a.j].is_wall = False
		maze[b.i][b.j].is_wall = False


	assert (rows % 2) and (cols % 2), 'Rows and cols must be odd'

	graph = [[MazeNode(y, x) for x in range(cols)] for y in range(rows)]

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
