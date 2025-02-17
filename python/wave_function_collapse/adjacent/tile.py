"""
Tile class for adjacent tiling WFC algorithm

Author: Sam Barba
Created 19/07/2023
"""


class Tile:
	def __init__(self, img):
		self.img = img
		self.neighbour_options = {'north': [], 'east': [], 'south': [], 'west': []}
