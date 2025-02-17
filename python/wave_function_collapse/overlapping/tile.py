"""
Tile class for overlapping tiling WFC algorithm

Author: Sam Barba
Created 15/02/2025
"""


class Tile:
	def __init__(self, img):
		self.img = img
		self.centre_pixel = img[img.shape[0] // 2, img.shape[1] // 2]
		self.neighbour_options = {'north': [], 'east': [], 'south': [], 'west': []}

	def generate_adjacency_rules(self, tiles):
		"""
		Populate possible N/E/S/W neighbour options by comparing the overlap of this tile's N/E/S/W edges with the
		opposite edges of other tiles
		"""

		north_colours = self.img[:-1]
		east_colours = self.img[:, 1:]
		south_colours = self.img[1:]
		west_colours = self.img[:, :-1]

		for idx, other in enumerate(tiles):
			other_north_colours = other.img[:-1]
			other_east_colours = other.img[:, 1:]
			other_south_colours = other.img[1:]
			other_west_colours = other.img[:, :-1]

			if (north_colours == other_south_colours).all():
				self.neighbour_options['north'].append(idx)
			if (east_colours == other_west_colours).all():
				self.neighbour_options['east'].append(idx)
			if (south_colours == other_north_colours).all():
				self.neighbour_options['south'].append(idx)
			if (west_colours == other_east_colours).all():
				self.neighbour_options['west'].append(idx)
