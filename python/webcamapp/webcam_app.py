"""
OpenCV webcam app for finding the closest matching colour
in a video to a specified target colour

Author: Sam Barba
Created 02/08/2022
"""

import cv2 as cv
import numpy as np

PROCESSING = ' /@@@@@@@     /@@@@@@@      /@@@@@@      /@@@@@@     /@@@@@@@@     /@@@@@@      /@@@@@@     /@@@@@@    /@@   /@@     /@@@@@@\n' \
	'| @@__  @@   | @@__  @@    /@@__  @@    /@@__  @@   | @@_____/    /@@__  @@    /@@__  @@   |_  @@_/   | @@@ | @@    /@@__  @@\n' \
	'| @@  \\ @@   | @@  \\ @@   | @@  \\ @@   | @@  \\__/   | @@         | @@  \\__/   | @@  \\__/     | @@     | @@@@| @@   | @@  \\__/\n' \
	'| @@@@@@@/   | @@@@@@@/   | @@  | @@   | @@         | @@@@@@     |  @@@@@@    |  @@@@@@      | @@     | @@ @@ @@   | @@ /@@@@\n' \
	'| @@____/    | @@__  @@   | @@  | @@   | @@         | @@___/      \\____  @@    \\____  @@     | @@     | @@  @@@@   | @@|_  @@\n' \
	'| @@         | @@  \\ @@   | @@  | @@   | @@    @@   | @@          /@@  \\ @@    /@@  \\ @@     | @@     | @@\\  @@@   | @@  \\ @@\n' \
	'| @@         | @@  | @@   |  @@@@@@/   |  @@@@@@/   | @@@@@@@@   |  @@@@@@/   |  @@@@@@/    /@@@@@@   | @@ \\  @@   |  @@@@@@/   /@@   /@@   /@@\n' \
	'|__/         |__/  |__/    \\______/     \\______/    |________/    \\______/     \\______/    |______/   |__/  \\__/    \\______/   |__/  |__/  |__/'

TARGET_RGB = (0, 0, 255)
MAX_DIST = 441.673  # Maximum Euclidean distance between 2 colours = root(255^2 * 3)
CROSSHAIR_GAP = 8
SCALE = 2

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def find_most_similar_colour(frame, target_colour):
	# Convert RGB to BGR, as OpenCV image data is BGR
	target_colour = target_colour[::-1]

	# Convert frame to a list of BGR values
	frame_reshaped = np.vstack(frame)

	# Find index of pixel with minimum Euclidean distance from target colour
	closest_idx = np.argmin([dist(px, target_colour) for px in frame_reshaped])

	# Convert list index back to coordinates
	y, x = np.unravel_index(closest_idx, frame.shape[:2])  # Row, column

	return x, y

def dist(colour1, colour2):
	"""Euclidean distance between 2 colours"""
	return np.linalg.norm(np.array(colour1) - np.array(colour2))

def crosshairs(frame, x, y):
	"""Draw red crosshairs (BGR = 0 0 255) to highlight a certain point in a frame"""

	frame[:y - CROSSHAIR_GAP, x] = \
		frame[y + CROSSHAIR_GAP:, x] = \
		frame[y, :x - CROSSHAIR_GAP] = \
		frame[y, x + CROSSHAIR_GAP:] = np.array([0, 0, 255])

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	vidcap = cv.VideoCapture(0)

	# Set width and height properties to small values for faster processing
	# (will be scaled up by SCALE when displayed)
	vidcap.set(3, 320)
	vidcap.set(4, 180)
	success, img = vidcap.read()

	while success:
		success, img = vidcap.read()

		# Coords and closest colour to target RGB
		x, y = find_most_similar_colour(img, TARGET_RGB)
		closest_colour = img[y][x]

		# Display the frame, scaled and with the closest matching pixel highlighted
		width = int(img.shape[1] * SCALE)
		height = int(img.shape[0] * SCALE)
		img = cv.resize(img, (width, height), interpolation=cv.INTER_LINEAR)
		crosshairs(img, int(x * SCALE), int(y * SCALE))

		# Also show frame of target colour and closest matching colour
		target_colour_frame = np.zeros((90, 90, 3))
		target_colour_frame[:] = np.array(TARGET_RGB[::-1])
		closest_colour_frame = np.zeros((90, 90, 3))
		closest_colour_frame[:] = closest_colour
		target_and_closest = np.hstack((target_colour_frame, closest_colour_frame))

		# Put target_and_closest in bottom-left of img
		img[-target_and_closest.shape[0]:, :target_and_closest.shape[1]] = target_and_closest
		percentage_match = (1 - dist(closest_colour[::-1], TARGET_RGB) / MAX_DIST) * 100
		cv.putText(img=img,
			text=f'% match: {percentage_match:.1f}',
			org=(11, img.shape[0] - 40),
			fontFace=cv.FONT_HERSHEY_SIMPLEX,
			fontScale=0.5,
			color=tuple(255 - c for c in TARGET_RGB)[::-1])

		# Show img
		cv.imshow("Video ('Q' to quit)", img)

		# 'Q' = quit
		if cv.waitKey(1) & 255 == ord('q'):
			break

	vidcap.release()
	cv.destroyAllWindows()

if __name__ == '__main__':
	print(PROCESSING)
	main()
