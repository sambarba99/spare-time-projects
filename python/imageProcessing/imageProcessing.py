# Image Processing
# Author: Sam Barba
# Created 23/09/2021

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
			# 220.8 = max distance from white / 2
			newPixel = (255, 255, 255) if d < 220.8 else (0, 0, 0)
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

	maxDist = 441.67
	print("Best RGB = {} {} {}  ({} % match)".format(rBest, gBest, bBest, round(100 * (1 - closestDist / maxDist), 2)))

	return newImg

# Euclidean distance between 2 colours
def dist(pixel, targetPixel):
	return sum((p - t) ** 2 for p, t in zip(pixel, targetPixel)) ** 0.5

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

imgs = [Image.open(src) for src in IMG_SRCS]
for i in range(len(imgs)):
	width, height = imgs[i].size

	if max(width, height) > MAX_SIZE:
		newWidth = MAX_SIZE if width > height else round(width / height * MAX_SIZE)
		newHeight = MAX_SIZE if height > width else round(height / width * MAX_SIZE)
		imgs[i] = imgs[i].resize((newWidth, newHeight))

while True:
	choice = input("Enter: 1 to create binary image,"
		+ "\n 2 to find the nearest pixel to a certain colour,"
		+ "\n or X to exit: ").upper()

	if choice == "1":
		for img in imgs:
			binaryImage(img).show()
	elif choice == "2":
		r, g, b = map(int, input("\nInput the target RGB: ").split())
		for img in imgs:
			nearestColour(img, r, g, b).show()
	elif choice == "X":
		break
	print()
