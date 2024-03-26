"""
Polynomial regressor class

Author: Sam Barba
Created 18/10/2023
"""

import numpy as np


class PolynomialRegressor:
	def __init__(self, degree):
		self.degree = degree
		self.theta = None  # Polynomial coefficients

	def fit(self, x, y):
		"""Ordinary Least Squares (OLS) solution (as opposed to gradient descent)"""

		x_poly = np.ones((x.shape[0], 1))

		for d in range(1, self.degree + 1):
			x_poly = np.c_[x_poly, x ** d]

		self.theta = np.linalg.inv(x_poly.T.dot(x_poly)).dot(x_poly.T).dot(y)

	def predict(self, x):
		x_poly = np.ones((x.shape[0], 1))

		for d in range(1, self.degree + 1):
			x_poly = np.c_[x_poly, x ** d]

		y_pred = x_poly.dot(self.theta)

		return y_pred
