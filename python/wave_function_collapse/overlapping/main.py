"""
Visualisation of Wave Function Collapse (overlapping tiling algorithm)

1. First, an input image from ./src_imgs is read (change via SRC_IMG_NAME).
2. Patches (tiles) of a predefined size are extracted from the image into a list. E.g. with a source image of size
20x20, this tile list would be 400 long.
3. Tile objects are created using these images, and adjacency rules are created by using the colours on the edges of
each tile (TILE_SIZE - 1 wide) and comparing the overlap with the edges of other tiles.
4. A grid of cells (Cell.py) is then initiated, each one initially having multiple possible states (Tile.py). This is
their superposition. By iterating WFC and checking which neighbour tile states are allowed, each cell will eventually
have just one state (its superposition is 'collapsed'), meaning its tile image can be visualised.

Author: Sam Barba
Created 15/02/2025
"""

import random
import sys

import cv2 as cv
import numpy as np
import pygame as pg

from cell import Cell
from tile import Tile


SRC_IMG_NAME = 'flowers'  # flowers, island, link, skyline, spirals, or waves
TILE_SIZE = 3
COLLAGE_WIDTH = 41  # In tiles
COLLAGE_HEIGHT = 25
CELL_SIZE = 20
MAX_DEPTH = 20
RENDER_ENTROPY_VALUES = True
FPS = 60


def generate_tiles():
	src_img = cv.imread(f'./src_imgs/{SRC_IMG_NAME}.png')
	src_img = cv.cvtColor(src_img, cv.COLOR_BGR2RGB)
	h, w, _ = src_img.shape

	# Create Tile objects by looping through all possible TILE_SIZE x TILE_SIZE tiles in the source image,
	# wrapping around edges

	tile_imgs = []
	for y in range(h):
		for x in range(w):
			y_indices = np.arange(y, y + TILE_SIZE) % h
			x_indices = np.arange(x, x + TILE_SIZE) % w
			tile = src_img[np.ix_(y_indices, x_indices)]
			tile_imgs.append(tile)
	tiles = [Tile(tile_img) for tile_img in tile_imgs]

	# Show the source image with a collage of all the extracted tiles

	new_h = src_img.shape[0] * 16
	new_w = src_img.shape[1] * 16
	src_img_zoomed = cv.resize(src_img, (new_w, new_h), interpolation=cv.INTER_NEAREST)
	cv.imshow(f'{SRC_IMG_NAME}.png ({w}x{h})', cv.cvtColor(src_img_zoomed, cv.COLOR_RGB2BGR))

	collage_h = h * TILE_SIZE + h - 1
	collage_w = w * TILE_SIZE + w - 1
	collage = np.full((collage_h, collage_w, 3), (13, 17, 23), dtype=np.uint8)
	for idx, tile_img in enumerate(tile_imgs):
		row = idx // w
		col = idx % w
		y = row * (TILE_SIZE + 1)  # Space tiles by 1px
		x = col * (TILE_SIZE + 1)
		collage[y:y + TILE_SIZE, x:x + TILE_SIZE] = tile_img

	collage_zoomed = cv.resize(collage, (collage_w * 8, collage_h * 8), interpolation=cv.INTER_NEAREST)
	cv.imshow(f'All {TILE_SIZE}x{TILE_SIZE} tiles', cv.cvtColor(collage_zoomed, cv.COLOR_RGB2BGR))
	cv.waitKey()
	cv.destroyAllWindows()

	# Generate adjacency rules

	for tile in tiles:
		tile.generate_adjacency_rules(tiles)

	return tiles


def wave_function_collapse(first_cell=False):
	# Choose a cell whose superposition to collapse to one state

	if first_cell:
		# If 'flowers' or 'skyline', start on the ground (bottom of grid). Otherwise, start in the grid centre.
		if SRC_IMG_NAME == 'flowers':
			next_cell_to_collapse = grid[-1][COLLAGE_WIDTH // 2]
			next_cell_to_collapse.tile_options = [next_cell_to_collapse.tile_options[336]]
		elif SRC_IMG_NAME == 'skyline':
			next_cell_to_collapse = grid[-1][COLLAGE_WIDTH // 2]
			next_cell_to_collapse.tile_options = [next_cell_to_collapse.tile_options[401]]
		else:
			next_cell_to_collapse = grid[COLLAGE_HEIGHT // 2][COLLAGE_WIDTH // 2]
	else:
		# From all the non-collapsed cells, choose the one with minimum entropy, breaking any ties randomly
		uncollapsed = [cell for row in grid for cell in row if not cell.is_collapsed]
		if uncollapsed:
			uncollapsed.sort(key=lambda cell: cell.entropy)
			cells_with_min_entropy = [cell for cell in uncollapsed if cell.entropy == uncollapsed[0].entropy]
			next_cell_to_collapse = random.choice(cells_with_min_entropy)
		else:
			# If all are collapsed (1 state option left), we're done
			print('All superpositions collapsed to 1 state, starting again')
			draw_collage()
			return 'all cells collapsed'

	contradiction = next_cell_to_collapse.observe()
	if contradiction:
		print('Reached contradiction, starting again')
		return 'contradiction'

	# Propagate adjacency rules to ensure only legal superpositions remain

	stack = [(next_cell_to_collapse, 0)]

	while stack:
		current_cell, depth = stack.pop()
		if depth > MAX_DEPTH:
			continue

		for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # N, E, S, W
			adj_y = current_cell.y + dy  # Adjacent cell coords
			adj_x = current_cell.x + dx
			if adj_y not in range(COLLAGE_HEIGHT) or adj_x not in range(COLLAGE_WIDTH):
				# Coords out of bounds
				continue

			adjacent_cell = grid[adj_y][adj_x]

			if (dy, dx) == (-1, 0): opposite_dir = 'south'
			elif (dy, dx) == (0, 1): opposite_dir = 'west'
			elif (dy, dx) == (1, 0): opposite_dir = 'north'
			else: opposite_dir = 'east'

			for adj_tile_idx in adjacent_cell.tile_options:
				adj_tile = tiles[adj_tile_idx]

				# Loop through all possible tiles (states) of current_cell and check if at least one is compatible with
				# adj_tile. If so, then adj_tile_idx is allowed in adjacent_cell's superposition.

				adj_tile_allowed = any(
					tile in adj_tile.neighbour_options[opposite_dir]
					for tile in current_cell.tile_options
				)

				if not adj_tile_allowed:
					# If adj_tile isn't allowed, remove it from adjacent_cell's superposition
					adjacent_cell.tile_options = [i for i in adjacent_cell.tile_options if i != adj_tile_idx]

					if adjacent_cell.entropy == 0:
						print('Reached contradiction, starting again')
						return 'contradiction'

					# Adjacent cell updated, so add it to the stack to update its neighbours etc.
					if (adjacent_cell, depth + 1) not in stack:
						stack.append((adjacent_cell, depth + 1))

	draw_collage()
	if first_cell:
		pg.time.delay(1000)

	return 'propagated rules'


def draw_collage():
	scene.fill('black')

	for row in grid:
		for cell in row:
			if cell.is_collapsed:
				tile_idx = cell.tile_options[0]
				tile_colour = tiles[tile_idx].centre_pixel
			else:
				tile_colour = np.mean([tiles[idx].centre_pixel for idx in cell.tile_options], axis=0)
			pg.draw.rect(scene, tile_colour, (cell.x * CELL_SIZE, cell.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
			if not cell.is_collapsed and RENDER_ENTROPY_VALUES:
				cell_lbl = font.render(str(cell.entropy), True, 255 - tile_colour)
				lbl_rect = cell_lbl.get_rect(
					center=((cell.x + 0.5) * CELL_SIZE, (cell.y + 0.5) * CELL_SIZE)
				)
				scene.blit(cell_lbl, lbl_rect)

	pg.display.update()
	clock.tick(FPS)

	for event in pg.event.get():
		if event.type == pg.QUIT:
			sys.exit()


if __name__ == '__main__':
	tiles = generate_tiles()

	pg.init()
	pg.display.set_caption('Wave Function Collapse (overlapping tiling algorithm)')
	scene = pg.display.set_mode((COLLAGE_WIDTH * CELL_SIZE, COLLAGE_HEIGHT * CELL_SIZE))
	font = pg.font.SysFont('consolas', 11)
	clock = pg.time.Clock()

	while True:
		grid = [[Cell(y, x, len(tiles)) for x in range(COLLAGE_WIDTH)] for y in range(COLLAGE_HEIGHT)]
		draw_collage()
		pg.time.delay(1000)
		_ = wave_function_collapse(first_cell=True)
		while True:
			result = wave_function_collapse()
			if result in ('contradiction', 'all cells collapsed'):
				break

		# if result == 'all cells collapsed':
		# 	pg.image.save(scene, f'../{SRC_IMG_NAME}.png')
		pg.time.delay(2000)
