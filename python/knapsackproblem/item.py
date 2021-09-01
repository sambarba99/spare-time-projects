"""
Item class for GA demo

Author: Sam Barba
Created 17/09/2021
"""

class Item:
	def __init__(self, index, value, weight):
		self.index = index
		self.value = value
		self.weight = weight

	def __repr__(self):
		return f'Item {self.index}:  value: {self.value}  weight: {self.weight}'
