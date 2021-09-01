# Voronoi diagram generator
# Author: Sam Barba
# Created 14/10/2021

from PIL import Image
import random
import sys

SIZE = 300

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def drawVoronoi(vertexCoords, colours):
	img = Image.new("RGB", (SIZE, SIZE))

	for x in range(SIZE):
		for y in range(SIZE):
			closestIdx = findClosest(x, y, vertexCoords)
			img.putpixel((x, y), tuple(colours[closestIdx]))

	return img

def findClosest(pixelX, pixelY, vertexCoords):
	closestIdx = 0
	closestDist = euclideanDist(pixelX, pixelY, *vertexCoords[closestIdx])

	for i in range(numVertices):
		d = euclideanDist(pixelX, pixelY, *vertexCoords[i])
		if d < closestDist:
			closestDist = d
			closestIdx = i

	return closestIdx

def euclideanDist(x1, y1, x2, y2):
	# Ignore square root for faster execution time
	return (x1 - x2) ** 2 + (y1 - y2) ** 2

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

while True:
	numVertices = int(input("How many vertices? "))
	vertexCoords = [[random.randint(5, SIZE - 5) for i in range(2)] for j in range(numVertices)]
	colours = [[random.randrange(256) for i in range(3)] for j in range(numVertices)]

	drawVoronoi(vertexCoords, colours).show()

	choice = input("\nEnter to continue or X to exit: ").upper()
	if len(choice) > 0 and choice[0] == 'X':
		break
	print()
