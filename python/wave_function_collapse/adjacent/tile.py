"""
Tile class for adjacent tiling WFC algorithm

Author: Sam Barba
Created 19/07/2023
"""


class Tile:
	def __init__(self, img, edge_colour_codes):
		self.img = img
		self.north_edge = edge_colour_codes[0]
		self.east_edge = edge_colour_codes[1]
		self.south_edge = edge_colour_codes[2]
		self.west_edge = edge_colour_codes[3]
		self.neighbour_options = {'north': [], 'east': [], 'south': [], 'west': []}

	def generate_adjacency_rules(self, tiles):
		"""
		Populate possible N/E/S/W neighbour options by comparing this tile's N/E/S/W edge colour codes to the flipped
		colour codes of the opposite edges of other tiles
		"""

		for idx, other in enumerate(tiles):
			if self.north_edge == other.south_edge[::-1]:
				self.neighbour_options['north'].append(idx)
			if self.east_edge == other.west_edge[::-1]:
				self.neighbour_options['east'].append(idx)
			if self.south_edge == other.north_edge[::-1]:
				self.neighbour_options['south'].append(idx)
			if self.west_edge == other.east_edge[::-1]:
				self.neighbour_options['west'].append(idx)
