"""
Circle class for generation.py

Author: Sam Barba
Created 23/09/2022
"""

class Circle:
	"""A circle represented by a complex centre point and a radius"""

	def __init__(self, x, y, r):
		self.centre = x + y * 1j
		self.r = r
		self.curvature = 1 / r

	def __repr__(self):
		return f'Circle (centre=({self.centre.real}, {self.centre.imag}), r={self.r})'
