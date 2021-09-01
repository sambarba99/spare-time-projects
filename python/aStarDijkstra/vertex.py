# Vertex for A* and Dijkstra demo
# Author: Sam Barba
# Created 20/09/2021

class Vertex:
	def __init__(self, x, y):
		self.x = x
		self.y = y

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
		# Usually other way around; this is for the sake of using x,y rather than y,x
		rows, cols = len(maze[0]), len(maze)

		if self.x - dist >= 0 and maze[self.x - dist][self.y].isWall == isWall:
			neighbours.append(maze[self.x - dist][self.y])
		if self.x + dist < cols and maze[self.x + dist][self.y].isWall == isWall:
			neighbours.append(maze[self.x + dist][self.y])
		if self.y - dist >= 0 and maze[self.x][self.y - dist].isWall == isWall:
			neighbours.append(maze[self.x][self.y - dist])
		if self.y + dist < rows and maze[self.x][self.y + dist].isWall == isWall:
			neighbours.append(maze[self.x][self.y + dist])

		return neighbours

	def getFcost(self):
		return self.gCost + self.hCost
