# Daedalus, the labyrinth creator
# Author: Sam Barba
# Created 20/09/2021

import random
from vertex import *

class Daedalus:
	def __init__(self, rows, cols):
		self.rows = rows
		self.cols = cols

	def makeMaze(self):
		maze = [[Vertex(x, y) for y in range(self.rows)] for x in range(self.cols)]

		# Make top-left start
		maze[0][0].isWall = False
		current = maze[0][0]
		stack = []

		while True:
			walls = current.getNeighbours(maze, True)

			if walls:
				next = random.choice(walls)
				stack.append(current)
				self.__removeWalls(maze, current, next)
				current = next
			elif stack:
				current = stack.pop()
			else:
				return maze

	def __removeWalls(self, maze, a, b):
		midX = (a.x + b.x) // 2
		midY = (a.y + b.y) // 2
		maze[midX][midY].isWall = False
		maze[a.x][a.y].isWall = False
		maze[b.x][b.y].isWall = False
