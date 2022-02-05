# A* and Dijkstra demo
# Author: Sam Barba
# Created 20/09/2021

from daedalus import Daedalus
import pygame as pg
from time import sleep

CELL_SIZE = 18
ROWS = 49
COLS = 99

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def aStar(maze, startVertex, targetVertex):
	openSet, closedSet = [startVertex], []

	while openSet:
		cheapestVertex = min(openSet, key=lambda v: (v.getFcost(), v.hCost))

		if cheapestVertex == targetVertex:
			return retracePath(targetVertex, startVertex)

		openSet.remove(cheapestVertex)
		closedSet.append(cheapestVertex)

		neighbours = cheapestVertex.getNeighbours(maze, False)
		for n in neighbours:
			if n in closedSet: continue

			costMoveToN = cheapestVertex.gCost + dist(cheapestVertex, n)
			if costMoveToN < n.gCost or n not in openSet:
				n.gCost = costMoveToN
				n.hCost = dist(n, targetVertex)
				n.parentVertex = cheapestVertex

				if n not in openSet:
					openSet.append(n)

# Manhattan distance
def dist(a, b):
	return abs(a.x - b.x) + abs(a.y - b.y)

# Dijkstra's algorithm for Shortest Path Tree
def dijkstra(maze, startVertex, targetVertex):
	unvisited = [vertex for row in maze for vertex in row if not vertex.isWall]

	# Costs nothing to get from start to start (startVertex parent will always be None)
	startVertex.cost = 0

	while unvisited:
		cheapestVertex = min(unvisited, key=lambda v: v.cost)

		neighbours = cheapestVertex.getNeighbours(maze, False)
		for n in neighbours:
			# Adjust cost and parent (weight between vertices = 1, i.e. 1 step needed)
			if cheapestVertex.cost + 1 < n.cost:
				n.cost = cheapestVertex.cost + 1
				n.parentVertex = cheapestVertex

		# Cheapest vertex has now been visited
		unvisited.remove(cheapestVertex)

	return retracePath(targetVertex, startVertex)

def retracePath(targetVertex, startVertex):
	# Trace back from end
	current = targetVertex
	path = [current]

	while current != startVertex:
		current = current.parentVertex
		path.append(current)

	return path[::-1]

def draw(scene, maze, path):
	for y in range(ROWS):
		for x in range(COLS):
			c = (0, 0, 0) if maze[y][x].isWall else (80, 80, 80)

			pg.draw.rect(scene, c, pg.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

	pg.display.flip()
	
	sleep(1)
	for v in path:
		pg.draw.rect(scene, (255, 0, 0), pg.Rect(v.x * CELL_SIZE, v.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
		sleep(0.01)
		pg.display.flip()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

mazeGenerator = Daedalus(ROWS, COLS)

pg.init()
pg.display.set_caption("A* and Dijkstra demo")
scene = pg.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE))

while True:
	maze = mazeGenerator.makeMaze()

	startVertex = maze[0][0]
	targetVertex = maze[ROWS - 1][COLS - 1]

	path = aStar(maze, startVertex, targetVertex)
	#path = dijkstra(maze, startVertex, targetVertex)

	draw(scene, maze, path)
	sleep(2)
