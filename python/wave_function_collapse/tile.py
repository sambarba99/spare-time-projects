"""
Tile class

Author: Sam Barba
Created 19/07/2023
"""

import pygame as pg


class Tile:
	def __init__(self, img, edge_colour_codes, num_rotations):
		self.img = img
		self.north_edge = edge_colour_codes[0]
		self.east_edge = edge_colour_codes[1]
		self.south_edge = edge_colour_codes[2]
		self.west_edge = edge_colour_codes[3]
		self.num_rotations = num_rotations
		self.north_options = []
		self.east_options = []
		self.south_options = []
		self.west_options = []

	def generate_adjacency_rules(self, tiles):
		"""
		Generate adjacency rules by comparing the N/E/S/W colour codes to the reversed
		ones of other tiles (reversed, because to check if they fit together, one can
		imagine rotating the other tile to try and make it fit with this tile - when
		rotated, the colour code of the opposite edge, e.g. N vs S, becomes reversed)
		"""

		for tile in tiles:
			if self.north_edge == tile.south_edge[::-1]:
				self.north_options.append(tile)
			if self.east_edge == tile.west_edge[::-1]:
				self.east_options.append(tile)
			if self.south_edge == tile.north_edge[::-1]:
				self.south_options.append(tile)
			if self.west_edge == tile.east_edge[::-1]:
				self.west_options.append(tile)

	def rotate(self, num_times):
		"""
		Rotate a tile clockwise a certain no. of times, including image and edge colours
		"""

		rotated_img = pg.transform.rotate(self.img, -90 * num_times)  # Negative for clockwise
		old_edges = [self.north_edge, self.east_edge, self.south_edge, self.west_edge]
		new_edges = [old_edges[(i - num_times) % 4] for i in range(4)]
		return Tile(rotated_img, new_edges, None)
