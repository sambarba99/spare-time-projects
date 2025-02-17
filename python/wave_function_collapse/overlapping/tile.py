"""
Tile class for overlapping WFC algorithm

Author: Sam Barba
Created 15/02/2025
"""


class Tile:
	def __init__(self, img):
		self.img = img
		self.centre_pixel = img[img.shape[0] // 2, img.shape[1] // 2]
		self.overlap_size = img.shape[0] // 2 + 1
		self.neighbour_options = {'north': [], 'east': [], 'south': [], 'west': []}

	def generate_adjacency_rules(self, tiles):
		"""Generate adjacency rules by comparing this tile's N/E/S/W colour codes to the ones of other tiles"""

		north_colours = self.img[:self.overlap_size]
		east_colours = self.img[:, -self.overlap_size:]
		south_colours = self.img[-self.overlap_size:]
		west_colours = self.img[:, :self.overlap_size]

		for other in tiles:
			if self is other:
				continue
			other_north_colours = other.img[:self.overlap_size]
			other_east_colours = other.img[:, -self.overlap_size:]
			other_south_colours = other.img[-self.overlap_size:]
			other_west_colours = other.img[:, :self.overlap_size]

			if (north_colours == other_south_colours).all():
				self.neighbour_options['north'].append(other)
			if (east_colours == other_west_colours).all():
				self.neighbour_options['east'].append(other)
			if (south_colours == other_north_colours).all():
				self.neighbour_options['south'].append(other)
			if (west_colours == other_east_colours).all():
				self.neighbour_options['west'].append(other)
