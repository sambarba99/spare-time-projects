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

# For displaying target colour vs closest match information
FONT = cv.FONT_HERSHEY_SIMPLEX
BOTTOM_LEFT = (5, 170)
FONT_SCALE = 0.7
FONT_COLOUR = tuple(255 - c for c in TARGET_RGB)[::-1]  # Reversed because OpenCV uses BGR

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
	cap = cv.VideoCapture(0)

	# Set width and height properties to small values for faster processing
	# (will be scaled up by SCALE when displayed)
	cap.set(3, 320)
	cap.set(4, 180)

	while True:
		# Capture video frame by frame
		_, frame = cap.read()

		# Coords and closest colour to target RGB
		x, y = find_most_similar_colour(frame, TARGET_RGB)
		closest_colour = frame[y][x]

		# Display the frame, scaled and with the closest matching pixel highlighted
		width = int(frame.shape[1] * SCALE)
		height = int(frame.shape[0] * SCALE)
		frame = cv.resize(frame, (width, height), interpolation=cv.INTER_LINEAR)
		crosshairs(frame, int(x * SCALE), int(y * SCALE))
		cv.imshow("Video ('Q' to quit)", frame)

		# Also show frame of target colour and closest matching colour
		target_colour_frame = np.zeros((180, 180, 3))
		target_colour_frame[:] = np.array(TARGET_RGB[::-1])
		closest_match_colour_frame = np.zeros((180, 180, 3))
		closest_match_colour_frame[:] = np.array(closest_colour)
		final_frame = np.hstack((target_colour_frame, closest_match_colour_frame)).astype(np.uint8)

		percentage_match = (1 - dist(closest_colour[::-1], TARGET_RGB) / MAX_DIST) * 100
		cv.putText(final_frame, f'% match: {percentage_match:.1f}', BOTTOM_LEFT, FONT, FONT_SCALE, FONT_COLOUR)
		cv.imshow('Target colour and closest match', final_frame)

		# 'Q' = quit
		if cv.waitKey(1) & 255 == ord('q'):
			break

	cap.release()
	cv.destroyAllWindows()

if __name__ == '__main__':
	print(PROCESSING)
	main()
