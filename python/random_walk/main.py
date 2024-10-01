"""
Random walk generator

Author: Sam Barba
Created 18/10/2024
"""

from time import perf_counter

from cv2 import imwrite
import numpy as np
from tqdm import tqdm


DIRECTION_DX_DY = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}


def random_walk(num_steps, border=10):
	x = y = 0
	path = np.zeros((num_steps + 1, 2), dtype=int)
	path[0] = [y, x]
	rand_directions = np.random.choice(['N', 'E', 'S', 'W'], size=num_steps)
	for idx, r in enumerate(tqdm(rand_directions, desc='Walking', ascii=True, unit='steps')):
		dx, dy = DIRECTION_DX_DY[r]
		x += dx
		y += dy
		path[idx] = [y, x]

	path[:, 0] -= path[:, 0].min()
	path[:, 1] -= path[:, 1].min()
	path_height = path[:, 0].max() + 1
	path_width = path[:, 1].max() + 1
	img_height = path_height + 2 * border
	img_width = path_width + 2 * border

	coord_visits = dict()
	for coord in tqdm(path, desc='Counting coord visits', ascii=True):
		coord_visits[tuple(coord)] = coord_visits.get(tuple(coord), 0) + 1
	max_visits = max(coord_visits.values())

	img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
	for y, x in tqdm(path, desc='Creating image', ascii=True):
		colour = round(20 + coord_visits[y, x] * (255 - 20) / max_visits)
		img[y + border, x + border] = (colour, colour, colour)
	img[*path[0] + border] = (0, 0, 255)
	imwrite('./images/rand_walk.png', img)


if __name__ == '__main__':
	start = perf_counter()
	random_walk(int(1e7))
	interval = perf_counter() - start

	print(f'\nCompleted in {interval:.3g}s')
