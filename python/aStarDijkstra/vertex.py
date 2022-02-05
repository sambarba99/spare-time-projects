# Vertex for A* and Dijkstra demo
# Author: Sam Barba
# Created 20/09/2021

class Vertex:
	def __init__(self, y, x): # Y before X, as 2D arrays are row-major
		self.y = y
		self.x = x

		# For A* and Dijkstra
		self.isWall = True
		self.parentVertex = None

		# For A* only
		self.gCost = 0
		self.hCost = 0

		# For Dijkstra only
		self.cost = 10 ** 9 # Inf

	def getNeighbours(self, maze, mazeGeneration):
		dist = 2 if mazeGeneration else 1
		isWall = mazeGeneration

		neighbours = []
		rows, cols = len(maze), len(maze[0])

		if self.x - dist >= 0 and maze[self.y][self.x - dist].isWall == isWall:
			neighbours.append(maze[self.y][self.x - dist])
		if self.x + dist < cols and maze[self.y][self.x + dist].isWall == isWall:
			neighbours.append(maze[self.y][self.x + dist])
		if self.y - dist >= 0 and maze[self.y - dist][self.x].isWall == isWall:
			neighbours.append(maze[self.y - dist][self.x])
		if self.y + dist < rows and maze[self.y + dist][self.x].isWall == isWall:
			neighbours.append(maze[self.y + dist][self.x])

		return neighbours

	def getFcost(self):
		return self.gCost + self.hCost
