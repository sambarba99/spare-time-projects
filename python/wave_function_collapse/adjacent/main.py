"""
Visualisation of Wave Function Collapse (adjacent tiling algorithm)

1. First, a set of tile images is loaded from ./tile_imgs (change via COLLAGE_TYPE).
2. Tile objects are created using these images.
3. Adjacency rules are created by using ./tile_imgs/COLLAGE_TYPE/_left_right_adjacencies.json.
4. A grid of cells (Cell.py) is then initiated, each one initially having multiple possible states (Tile.py). This is
their superposition. By iterating WFC and checking which neighbour tile states are allowed, each cell will eventually
have just one state (its superposition is 'collapsed'), meaning its tile image can be visualised.

Author: Sam Barba
Created 19/07/2023
"""

import glob
import json
import random
import re
import sys

import pygame as pg

from cell import Cell
from tile import Tile


COLLAGE_TYPE = 'circuit'  # castle, circuit, or water
COLLAGE_WIDTH = 49  # In tiles
COLLAGE_HEIGHT = 29
RENDER_ENTROPY_VALUES = True
FPS = 60


def generate_tiles():
	# Create Tile objects by reading image files

	tiles, tile_names = [], []
	tile_name_to_orientations = dict()
	for img_path in glob.iglob(f'./tile_imgs/{COLLAGE_TYPE}/*.png'):
		tile = Tile(pg.image.load(img_path))
		tiles.append(tile)

		# Tile file names have the format <name_of_tile_type>_<orientations>.png
		file_name = img_path.split('\\')[-1].removesuffix('.png')
		match = re.match(r'^(.*)_(\d)$', file_name)
		tile_type = match.group(1)
		num_orientations = int(match.group(2))

		# Original file is the tile in its first orientation (orientation number 0)
		tile_name = f'{tile_type} 0'
		tile_names.append(tile_name)
		tile_name_to_orientations[tile_name] = num_orientations

		if num_orientations > 1:
			# If tile has multiple orientations, rotate it to get all of them
			for n in range(1, num_orientations):
				rotated_img = pg.transform.rotate(tile.img, -90 * n)  # Negative for clockwise
				tiles.append(Tile(rotated_img))
				tile_name = f'{tile_type} {n}'
				tile_names.append(tile_name)
				tile_name_to_orientations[tile_name] = num_orientations

	# Populate each tile's north/east/south/west neighbour options by using the adjacency list of valid left-right
	# (W-E) pairs, and inferring valid N-S, S-N, and E-W pairs

	with open(f'./tile_imgs/{COLLAGE_TYPE}/_left_right_adjacencies.json', 'r') as file:
		adjacency_list = json.load(file)

	for west_tile_name, east_tile_options in adjacency_list.items():
		# Add eastern (right) tiles to the left tile's eastern neighbour options
		# (i.e. what's already in the JSON)
		w_tile_idx = tile_names.index(west_tile_name)
		tiles[w_tile_idx].neighbour_options['east'] = [tile_names.index(i) for i in east_tile_options]

		# Rotate west tile and its east options (W-E pairs) clockwise
		# 1 rotation: W-E -> N-S
		# 2 rotations: W-E -> E-W
		# 3 rotations: W-E -> S-N
		w_tile_type, w_tile_orientation_num = west_tile_name.split()
		w_tile_orientations = tile_name_to_orientations[west_tile_name]
		for num_rots, new_dir in zip((1, 2, 3), ('south', 'west', 'north')):
			w_tile_rotated_name = f'{w_tile_type} {(int(w_tile_orientation_num) + num_rots) % w_tile_orientations}'
			w_tile_rotated_idx = tile_names.index(w_tile_rotated_name)
			for e_tile_name in east_tile_options:
				e_tile_type, e_tile_orientation_num = e_tile_name.split()
				e_tile_orientations = tile_name_to_orientations[e_tile_name]
				e_tile_rotated_name = f'{e_tile_type} {(int(e_tile_orientation_num) + num_rots) % e_tile_orientations}'
				e_tile_rotated_idx = tile_names.index(e_tile_rotated_name)
				tiles[w_tile_rotated_idx].neighbour_options[new_dir].append(e_tile_rotated_idx)

	for tile, name in zip(tiles, tile_names):
		print('\nTile:', name)
		for dir_ in ('north', 'east', 'south', 'west'):
			print(f'{dir_} options:', [tile_names[i] for i in tile.neighbour_options[dir_]])
			assert len(tile.neighbour_options[dir_]) == len(set(tile.neighbour_options[dir_]))

	return tiles


def wave_function_collapse(first_cell=False):
	# Choose a cell whose superposition to collapse to one state

	if first_cell:
		next_cell_to_collapse = grid[COLLAGE_HEIGHT // 2][COLLAGE_WIDTH // 2]  # Start in the grid centre
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

	stack = [next_cell_to_collapse]

	while stack:
		current_cell = stack.pop()

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
					if adjacent_cell not in stack:
						stack.append(adjacent_cell)

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
				tile_img = tiles[tile_idx].img
				img_rect = tile_img.get_rect(center=((cell.x + 0.5) * tile_size, (cell.y + 0.5) * tile_size))
				scene.blit(tile_img, img_rect)
			elif RENDER_ENTROPY_VALUES:
				cell_lbl = font.render(str(cell.entropy), True, 'white')
				lbl_rect = cell_lbl.get_rect(center=((cell.x + 0.5) * tile_size, (cell.y + 0.5) * tile_size))
				scene.blit(cell_lbl, lbl_rect)

	pg.display.update()
	clock.tick(FPS)

	for event in pg.event.get():
		if event.type == pg.QUIT:
			sys.exit()


if __name__ == '__main__':
	tiles = generate_tiles()
	tile_size = tiles[0].img.get_size()[0]

	pg.init()
	pg.display.set_caption('Wave Function Collapse (adjacent tiling algorithm)')
	scene = pg.display.set_mode((COLLAGE_WIDTH * tile_size, COLLAGE_HEIGHT * tile_size))
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
		# 	pg.image.save(scene, f'../{COLLAGE_TYPE}.png')
		pg.time.delay(2000)
