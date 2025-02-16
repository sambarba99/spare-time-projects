"""
Visualisation of Wave Function Collapse

Consists of a grid of cells (Cell.py), each one initially having multiple possible states (Tile.py).
This is their superposition. By iterating WFC, each cell will eventually have just 1 state/tile (its
superposition is 'collapsed'), meaning its image can be visualised.

All tile images in ./tile_imgs are named as:

<name>_<north left/middle/right colours>_<east left/middle/right colours>_<etc>_<no. times to rotate>.png.

E.g. ./tile_imgs/plain/bar_aaa_aba_aaa_aba_1.png.

Colours of their sides are read clockwise going around the edge of the tile. The number at the end is
how many times the tile needs to be rotated to generate all possible orientations (2 in total in the
example). Adjacency rules and all possible orientations are generated using these filename suffixes.

Author: Sam Barba
Created 19/07/2023
"""

import glob
import random
import sys

import pygame as pg

from cell import Cell
from tile import Tile


COLLAGE_TYPE = 'circuit'  # 'plain' or 'circuit'
COLLAGE_WIDTH = 80  # In tiles
COLLAGE_HEIGHT = 50
TILE_PX_SIZE = 15  # All images are 15px x 15px

tiles = grid = scene = font = None


def setup():
	assert COLLAGE_TYPE in ('plain', 'circuit')

	global tiles, scene, font

	pg.init()
	pg.display.set_caption('Visualisation of Wave Function Collapse')
	scene = pg.display.set_mode((TILE_PX_SIZE * COLLAGE_WIDTH, TILE_PX_SIZE * COLLAGE_HEIGHT))
	font = pg.font.SysFont('consolas', 11)

	# Read files in tile_imgs; decode their names to get colour codes and rotations

	tiles = []
	for img_path in glob.glob(f'./tile_imgs/{COLLAGE_TYPE}/*.png'):
		img = pg.image.load(img_path)
		split = img_path.split('/')[-1].split('_')
		edge_colour_codes = split[1:-1]
		num_rotations = split[-1].removesuffix('.png')
		tiles.append(Tile(img, edge_colour_codes, int(num_rotations)))

	# Generate all possible tile orientations

	rotated_tiles = []
	for tile in tiles:
		if tile.num_rotations:
			new_orientations = [tile.rotate(i) for i in range(1, tile.num_rotations + 1)]
			rotated_tiles.extend(new_orientations)
	tiles.extend(rotated_tiles)

	# Generate adjacency rules using edge colour codes

	for tile in tiles:
		tile.generate_adjacency_rules(tiles)


def wave_function_collapse(first_cell=False):
	global grid

	def choose_min_entropy_cell():
		"""
		Of all the non-collapsed cells, choose the one with minimum entropy (num. options).
		Break any ties randomly.
		"""

		uncollapsed = [cell for row in grid for cell in row if not cell.is_collapsed]

		# If all are collapsed (1 state option left), we're done
		if not uncollapsed:
			return None

		uncollapsed.sort(key=lambda cell: cell.entropy)
		min_entropy = uncollapsed[0].entropy
		cells_with_min_entropy = [cell for cell in uncollapsed if cell.entropy == min_entropy]

		return random.choice(cells_with_min_entropy)


	# Choose a cell whose superposition to collapse to one state

	next_cell_to_collapse = grid[COLLAGE_HEIGHT // 2][COLLAGE_WIDTH // 2] if first_cell else choose_min_entropy_cell()

	if next_cell_to_collapse:
		contradiction = next_cell_to_collapse.observe()
		if contradiction:
			print('Reached contradiction, starting again')
			grid = None
			return
	else:
		print('All cells collapsed to 1 state, starting again')
		grid = None
		return

	if first_cell:
		update_collage()
		pg.time.delay(1000)

	# Propagate adjacency rules to ensure only legal superpositions remain

	stack = [next_cell_to_collapse]

	while stack:
		current_cell = stack.pop()

		for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # N, E, S, W
			adj_x = current_cell.x + dx  # Adjacent cell coords
			adj_y = current_cell.y + dy

			if adj_x not in range(COLLAGE_WIDTH) or adj_y not in range(COLLAGE_HEIGHT):
				# Coords out of bounds
				continue

			adjacent_cell = grid[adj_y][adj_x]

			if (dx, dy) == (0, -1): opposite_dir = 'south'
			elif (dx, dy) == (1, 0): opposite_dir = 'west'
			elif (dx, dy) == (0, 1): opposite_dir = 'north'
			else: opposite_dir = 'east'

			for adj_tile in adjacent_cell.tile_options:
				# Loop through all possible tiles (states) of current_cell and check if at least one is
				# compatible with adj_tile. If so, then adj_tile is allowed in adjacent_cell's superposition.

				adj_tile_allowed = any(
					tile in getattr(adj_tile, f'{opposite_dir}_options')
					for tile in current_cell.tile_options
				)

				if not adj_tile_allowed:
					# If adj_tile isn't allowed, remove it from adjacent_cell's superposition
					adjacent_cell.tile_options = [t for t in adjacent_cell.tile_options if t != adj_tile]

					if adjacent_cell.entropy == 0:
						print('Reached contradiction, starting again')
						grid = None
						return

					if adjacent_cell not in stack:
						# Adjacent cell updated, so add it to the stack to update its neighbours etc.
						stack.append(adjacent_cell)


def update_collage():
	scene.fill('black')

	for row in grid:
		for cell in row:
			if cell.is_collapsed:
				tile_img = cell.tile_options[0].img
				tile_img_rect = tile_img.get_rect(
					center=((cell.x + 0.5) * TILE_PX_SIZE - 1, (cell.y + 0.5) * TILE_PX_SIZE - 1)
				)
				scene.blit(tile_img, tile_img_rect)
			# else:
			# 	cell_lbl = font.render(str(cell.entropy), True, 'white')
			# 	lbl_rect = cell_lbl.get_rect(
			# 		center=((cell.x + 0.5) * TILE_PX_SIZE - 1, (cell.y + 0.5) * TILE_PX_SIZE - 1)
			# 	)
			# 	scene.blit(cell_lbl, lbl_rect)

	pg.display.update()

	for event in pg.event.get():
		if event.type == pg.QUIT:
			sys.exit()


if __name__ == '__main__':
	setup()

	while True:
		grid = [[Cell(x, y, tiles) for x in range(COLLAGE_WIDTH)] for y in range(COLLAGE_HEIGHT)]

		wave_function_collapse(first_cell=True)

		while True:
			wave_function_collapse()
			if grid:
				update_collage()
			else:
				break
		# pg.image.save(scene, 'output.png')
		pg.time.delay(2000)
