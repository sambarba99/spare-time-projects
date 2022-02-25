# Image Processing
# Author: Sam Barba
# Created 23/09/2021

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMG_SRCS = [f"test{i}.jpg" for i in range(1, 5)]
MAX_SIZE = 600

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def binary_image(img):
	width, height = img.size

	new_img = Image.new("RGB", (width, height))

	for x in range(width):
		for y in range(height):
			d = dist(img.getpixel((x, y)), (255, 255, 255))
			# 97537.5 = max distance from white / 2 (ignoring square root)
			new_pixel = (255, 255, 255) if d < 97537.5 else (0, 0, 0)
			new_img.putpixel((x, y), new_pixel)

	return new_img

def nearest_colour(img, r_target, g_target, b_target):
	width, height = img.size

	closest_dist = dist(img.getpixel((0, 0)), (r_target, g_target, b_target))
	r_best, g_best, b_best = img.getpixel((0, 0))
	x_best = y_best = 0

	for x in range(width):
		for y in range(height):
			d = dist(img.getpixel((x, y)), (r_target, g_target, b_target))

			if d < closest_dist:
				closest_dist = d
				r_best, g_best, b_best = img.getpixel((x, y))
				x_best, y_best = x, y

	new_img = img.copy()
	for x in range(width):
		if abs(x - x_best) > 3:
			new_img.putpixel((x, y_best), (255, 0, 0))
	for y in range(height):
		if abs(y - y_best) > 3:
			new_img.putpixel((x_best, y), (255, 0, 0))

	max_dist = 195075
	percentage_match = 100 * (1 - (closest_dist / max_dist) ** 0.5)
	print(f"Best RGB = {r_best} {g_best} {b_best}  ({percentage_match:.2f} % match)")

	return new_img

def plot_histogram(img, idx):
	r, g, b = np.array(img.getdata()).T
	r_count = np.bincount(r, minlength=256)
	g_count = np.bincount(g, minlength=256)
	b_count = np.bincount(b, minlength=256)
	x_plot = list(range(256))

	plt.figure(figsize=(8, 6))
	plt.plot(x_plot, r_count, color="#ff0000")
	plt.plot(x_plot, g_count, color="#008000")
	plt.plot(x_plot, b_count, color="#0000ff")
	plt.legend(["R", "G", "B"])
	plt.xlabel("RGB value")
	plt.ylabel("Count")
	plt.title(f"Histogram for img {idx}")
	plt.show()

# Euclidean distance between 2 colours
def dist(pixel, target_pixel):
	pixel = np.array(pixel)
	target_pixel = np.array(target_pixel)
	# Ignore square root for faster execution
	return ((pixel - target_pixel) ** 2).sum()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

imgs = [Image.open(src) for src in IMG_SRCS]
for idx, img in enumerate(imgs):
	width, height = img.size

	if max(width, height) > MAX_SIZE:
		new_width = MAX_SIZE if width > height else round(width / height * MAX_SIZE)
		new_height = MAX_SIZE if height > width else round(height / width * MAX_SIZE)
		imgs[idx] = img.resize((new_width, new_height))

choice = input("Enter 1 to create binary image"
	+ "\nor 2 to find the nearest pixel to a certain colour: ")

if choice == "1":
	for idx, img in enumerate(imgs):
		img.show()
		plot_histogram(img, idx + 1)
		binary_image(img).show()
else:
	r, g, b = map(int, input("\nInput the target RGB: ").split())
	for idx, img in enumerate(imgs):
		img.show()
		plot_histogram(img, idx + 1)
		nearest_colour(img, r, g, b).show()
