"""
Cell class

Author: Sam Barba
Created 19/07/2023
"""

import random


class Cell:
	def __init__(self, x, y, tile_options):
		self.x = x
		self.y = y
		self.tile_options = tile_options

	@property
	def entropy(self):
		return len(self.tile_options)

	@property
	def is_collapsed(self):
		return self.entropy == 1

	def observe(self):
		"""
		By observing a cell, we collapse it into 1 possible state.
		A 'contradiction' boolean is returned (true if surrounding
		tiles mean that there are no options left, false otherwise)
		"""

		if self.entropy > 0:
			self.tile_options = [random.choice(self.tile_options)]
			return False

		return True
