"""
Tile class for adjacent tiling WFC algorithm

Author: Sam Barba
Created 19/07/2023
"""


class Tile:
	def __init__(self, img, edge_colour_codes, num_rotations=None):
		self.img = img
		self.north_edge = edge_colour_codes[0]
		self.east_edge = edge_colour_codes[1]
		self.south_edge = edge_colour_codes[2]
		self.west_edge = edge_colour_codes[3]
		self.num_rotations = num_rotations
		self.neighbour_options = {'north': [], 'east': [], 'south': [], 'west': []}

	def generate_adjacency_rules(self, tiles):
		"""
		Populate possible N/E/S/W neighbour options by comparing this tile's N/E/S/W edge colour codes to the flipped
		colour codes of the opposite edges of other tiles
		"""

		for other in tiles:
			if self.north_edge == other.south_edge[::-1]:
				self.neighbour_options['north'].append(other)
			if self.east_edge == other.west_edge[::-1]:
				self.neighbour_options['east'].append(other)
			if self.south_edge == other.north_edge[::-1]:
				self.neighbour_options['south'].append(other)
			if self.west_edge == other.east_edge[::-1]:
				self.neighbour_options['west'].append(other)
