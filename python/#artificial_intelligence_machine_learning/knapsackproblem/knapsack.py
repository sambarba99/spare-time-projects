"""
Knapsack class for GA demo

Author: Sam Barba
Created 17/09/2021
"""


class Knapsack:
	def __init__(self, item_config):
		# E.g. [False, True, True, False] means include 2nd and 3rd items
		self.item_config = item_config
		self.fitness = None


	def calc_fitness(self, all_items, N_ITEMS, KNAPSACK_CAPACITY):
		self.fitness = self.total_value(all_items, N_ITEMS) \
			if self.total_weight(all_items, N_ITEMS) <= KNAPSACK_CAPACITY else 0


	def total_value(self, all_items, N_ITEMS):
		return sum(all_items[i].value for i in range(N_ITEMS) if self.item_config[i])


	def total_weight(self, all_items, N_ITEMS):
		return sum(all_items[i].weight for i in range(N_ITEMS) if self.item_config[i])
