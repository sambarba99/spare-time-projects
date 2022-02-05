# Daedalus, the labyrinth creator
# Author: Sam Barba
# Created 20/09/2021

import random
from vertex import Vertex

class Daedalus:
	def __init__(self, rows, cols):
		self.rows = rows
		self.cols = cols

	def makeMaze(self):
		maze = [[Vertex(y, x) for x in range(self.cols)] for y in range(self.rows)]

		# Make top-left start
		maze[0][0].isWall = False
		current = maze[0][0]
		stack = []

		while True:
			walls = current.getNeighbours(maze, True)

			if walls:
				nextV = random.choice(walls)
				stack.append(current)
				self.__removeWalls(maze, current, nextV)
				current = nextV
			elif stack:
				current = stack.pop()
			else:
				return maze

	def __removeWalls(self, maze, a, b):
		midY = (a.y + b.y) // 2
		midX = (a.x + b.x) // 2
		maze[midY][midX].isWall = False
		maze[a.y][a.x].isWall = False
		maze[b.y][b.x].isWall = False
