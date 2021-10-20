# Image Processing
# Author: Sam Barba
# Created 23/09/2021

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMG_SRCS = ["test" + str(i + 1) + ".jpg" for i in range(7)]
MAX_SIZE = 600

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def binaryImage(img):
	width, height = img.size

	newImg = Image.new("RGB", (width, height))

	for x in range(width):
		for y in range(height):
			d = dist(img.getpixel((x, y)), (255, 255, 255))
			# 97537.5 = max distance from white / 2 (ignoring square root)
			newPixel = (255, 255, 255) if d < 97537.5 else (0, 0, 0)
			newImg.putpixel((x, y), newPixel)

	return newImg

def nearestColour(img, rTarget, gTarget, bTarget):
	width, height = img.size

	closestDist = dist(img.getpixel((0, 0)), (rTarget, gTarget, bTarget))
	rBest, gBest, bBest = img.getpixel((0, 0))
	xBest = yBest = 0

	for x in range(width):
		for y in range(height):
			d = dist(img.getpixel((x, y)), (rTarget, gTarget, bTarget))

			if d < closestDist:
				closestDist = d
				rBest, gBest, bBest = img.getpixel((x, y))
				xBest, yBest = x, y

	newImg = img.copy()
	for x in range(width):
		if abs(x - xBest) > 3:
			newImg.putpixel((x, yBest), (255, 0, 0))
	for y in range(height):
		if abs(y - yBest) > 3:
			newImg.putpixel((xBest, y), (255, 0, 0))

	maxDist = 195075
	percentageMatch = round(100 * (1 - (closestDist / maxDist) ** 0.5), 2)
	print(f"Best RGB = {rBest} {gBest} {bBest}  ({percentageMatch} % match)")

	return newImg

def plotHistogram(img, idx):
	imgData = np.array(img.getdata())
	r, g, b = imgData.T
	rCount = np.bincount(r, minlength=256)
	gCount = np.bincount(g, minlength=256)
	bCount = np.bincount(b, minlength=256)
	xPlot = list(range(256))

	plt.figure(figsize=(8, 6))
	plt.plot(xPlot, rCount, color="#ff0000")
	plt.plot(xPlot, gCount, color="#008000")
	plt.plot(xPlot, bCount, color="#0000ff")
	plt.legend(["R", "G", "B"])
	plt.xlabel("RGB value")
	plt.ylabel("Count")
	plt.title(f"Histogram for img {idx}")
	plt.show()

# Euclidean distance between 2 colours
def dist(pixel, targetPixel):
	pixel = np.array(pixel)
	targetPixel = np.array(targetPixel)
	# Ignore square root for faster execution
	return ((pixel - targetPixel) ** 2).sum()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

imgs = [Image.open(src) for src in IMG_SRCS]
for idx, img in enumerate(imgs):
	width, height = img.size

	if max(width, height) > MAX_SIZE:
		newWidth = MAX_SIZE if width > height else round(width / height * MAX_SIZE)
		newHeight = MAX_SIZE if height > width else round(height / width * MAX_SIZE)
		imgs[idx] = img.resize((newWidth, newHeight))

choice = input("Enter 1 to create binary image"
	+ "\nor 2 to find the nearest pixel to a certain colour: ")

if choice == "1":
	for idx, img in enumerate(imgs):
		img.show()
		plotHistogram(img, idx)
		binaryImage(img).show()
else:
	r, g, b = map(int, input("\nInput the target RGB: ").split())
	for idx, img in enumerate(imgs):
		img.show()
		plotHistogram(img, idx)
		nearestColour(img, r, g, b).show()
