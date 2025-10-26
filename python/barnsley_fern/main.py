"""
Barnsley fern generator

Author: Sam Barba
Created 26/10/2025
"""

import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


F1 = lambda x, y: (0, 0.16 * y)
F2 = lambda x, y: (0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.6)
F3 = lambda x, y: (0.2 * x - 0.26 * y, 0.23 * x + 0.22 * y + 1.6)
F4 = lambda x, y: (-0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44)

TRANSFORMS = [F1, F2, F3, F4]
PROBS = [0.01, 0.85, 0.07, 0.07]
NUM_STEPS = int(5e5)
IMG_SIZE = 1000


if __name__ == '__main__':
	sequence = random.choices([0, 1, 2, 3], weights=PROBS, k=NUM_STEPS)

	points = [(0, 0)]  # Start at origin
	for i in sequence:
		transform = TRANSFORMS[i]
		x, y = points[-1]
		next_x, next_y = transform(x, y)
		points.append((next_x, next_y))

	# Plot result
	x, y = zip(*points)
	plt.scatter(x, y, s=0.02, alpha=0.2, color='#008000')
	plt.axis('scaled')
	plt.show()

	# Make image out of result
	points = np.array(points)
	points[:, 0] -= points[:, 0].min()  # Shift x-coords so none are negative
	mins = points.min(axis=0)
	maxs = points.max(axis=0)
	size = maxs - mins
	scale_factor = (IMG_SIZE - 1) / max(size)
	scaled = (points - mins) * scale_factor
	scaled = scaled.astype(int)
	img_width = scaled[:, 0].max() - scaled[:, 0].min() + 1
	img_height = scaled[:, 1].max() - scaled[:, 1].min() + 1
	img = np.full((img_height, img_width, 3), (255, 255, 255), dtype=np.uint8)
	img[scaled[:, 1], scaled[:, 0]] = (0, 128, 0)  # Set fern points to green
	img = np.flipud(img)  # Flip image (y-axis)
	cv.imwrite('./fern.png', img)
