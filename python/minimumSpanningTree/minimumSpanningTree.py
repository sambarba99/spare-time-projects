# Minimum Spanning Tree demo
# Author: Sam Barba
# Created 17/09/2021

# Left-click: Add a vertex
# Right-click: Reset graph

import pygame as pg
import random
import sys

SIZE = 800
FPS = 20

# ---------------------------------------------------------------------------------------------------- #
# ---------------------------------------------  CLASSES  -------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

class Vertex:
	def __init__(self, idx, x, y, xVel, yVel):
		self.idx = idx
		self.x = x
		self.y = y
		self.xVel = xVel
		self.yVel = yVel

	def euclideanDist(self, other):
		# Ignore square root for faster execution
		return (self.x - other.x) ** 2 + (self.y - other.y) ** 2

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Prim's algorithm
def mst(graph):
	outTree = graph[:] # Initially set all vertices as out of tree
	inTree = []
	mstParents = [None] * len(graph)

	inTree.append(outTree.pop(0)) # Vertex 0 (arbitrary start) is first in tree

	while outTree:
		nearestIn = inTree[0]
		nearestOut = outTree[0]
		minDist = nearestIn.euclideanDist(nearestOut)

		# Find the nearest outside vertex to tree
		for vIn in inTree:
			for vOut in outTree:
				dist = vIn.euclideanDist(vOut)

				if dist < minDist:
					minDist = dist
					nearestOut = vOut
					nearestIn = vIn

		mstParents[nearestOut.idx] = nearestIn.idx

		inTree.append(nearestOut)
		outTree.remove(nearestOut)

	return mstParents

def drawMST(graph):
	if not graph: return

	scene.fill((20, 20, 20))
	mstParents = mst(graph)

	for i in range(1, len(graph)): # Start from 1 because mstParents[0] is None
		start = (graph[i].x, graph[i].y)
		end = (graph[mstParents[i]].x, graph[mstParents[i]].y)
		pg.draw.line(scene, (220, 220, 220), start, end)

	for v in graph:
		pg.draw.circle(scene, (230, 20, 20), (v.x, v.y), 5)

	pg.display.flip()

def movePoints(graph):
	for v in graph:
		v.x += v.xVel
		v.y += v.yVel

		while v.x < 5 or v.x > SIZE - 5:
			v.xVel = -v.xVel
			v.x += v.xVel
		while v.y < 5 or v.y > SIZE - 5:
			v.yVel = -v.yVel
			v.y += v.yVel

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

graph = []

pg.init()
pg.display.set_caption("Minimum Spanning Tree")
scene = pg.display.set_mode((SIZE, SIZE))
scene.fill((20, 20, 20))
pg.display.flip()
clock = pg.time.Clock()

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)
		elif event.type == pg.MOUSEBUTTONDOWN:
			if event.button == 1: # Left-click
				if len(graph) == 30:
					print("Size limit reached")
					continue

				x, y = event.pos
				# Constrain x and y to range [5, SIZE - 5]
				x = max(min(x, SIZE - 5), 5)
				y = max(min(y, SIZE - 5), 5)

				xVel, yVel = random.uniform(-3, 3), random.uniform(-3, 3)
				graph.append(Vertex(len(graph), x, y, xVel, yVel))

			elif event.button == 3: # Right-click
				graph = []
				scene.fill((20, 20, 20))
				pg.display.flip()

	drawMST(graph)
	movePoints(graph)
	clock.tick(FPS)
