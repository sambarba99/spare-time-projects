"""
Item class for GA demo

Author: Sam Barba
Created 17/09/2021
"""

class Item:
	def __init__(self, index, weight, value):
		self.index = index
		self.weight = weight
		self.value = value


	def __repr__(self):
		return f'Item {self.index}:  weight: {self.weight}  value: {self.value}'
