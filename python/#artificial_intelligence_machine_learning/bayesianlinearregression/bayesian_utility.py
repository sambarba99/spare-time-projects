"""
Utility class for Bayesian linear regression demo

Author: Sam Barba
Created 03/03/2022
"""

import numpy as np


class DataGenerator:
	"""Generate data for prediction modelling (a sine wave with a blank region)"""

	def __init__(self, noise):
		self.x_min = 0
		self.x_max = 2 * np.pi
		self.noise = noise


	def get_data(self, data_name, n):
		if data_name in ('TRAIN', 'VALIDATION'):
			return self.__make_data(n)
		if data_name == 'TEST':
			return self.__make_test_data(n)


	def __make_data(self, n):
		"""Make 2 sine wave portions"""

		portion_points1 = [self.x_min, self.x_max * 0.25]  # Start and end of portion 1
		portion_points2 = [self.x_max * 0.65, self.x_max]  # Start and end of portion 2
		points_per_portion = n // 2

		x = np.zeros(0)
		for start, end in (portion_points1, portion_points2):
			xi = np.linspace(start, end, points_per_portion)
			x = np.append(x, xi)

		x = x.reshape((n, 1))
		y = np.sin(x) + np.sin(2 * x)

		if self.noise > 0:
			y += np.random.normal(0, self.noise, size=(n, 1))

		return x, y


	def __make_test_data(self, n):
		# Full sin wave
		x = np.linspace(self.x_min, self.x_max, n).reshape((n, 1))
		y = np.sin(x) + np.sin(2 * x)
		return x, y


class RBFGenerator:
	"""Generate Gaussian RBF matrix"""

	def __init__(self, centres, radius, bias=True):
		self.m = len(centres)
		self.centres = centres.reshape((self.m, 1))
		self.r = radius
		self.bias = bias


	def evaluate(self, x):
		n = len(x)

		distances = [[sum((coord_x - coord_centre) ** 2) for coord_centre in self.centres] for coord_x in x]
		phi = np.exp(-np.array(distances) / self.r ** 2)

		if self.bias:
			phi = np.hstack((np.ones((n, 1)), phi))

		return phi


def mae(pred, actual):
	"""Mean absolute error"""

	return np.abs(pred - actual).sum() / len(actual)
