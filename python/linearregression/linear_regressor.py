"""
Linear regressor for linear_regression.py

Author: Sam Barba
Created 10/11/2021
"""

import numpy as np

class LinearRegressor:
	def __init__(self):
		self.x_train = None
		self.y_train = None
		self.weights = None
		self.bias = 0
		self.cost_history = []

	def fit(self, x_train, y_train, learning_rate=1e-4, converge_threshold=1e-9):
		"""Gradient descent"""

		def calculate_gradients(weights, bias):
			y_predictions = self.x_train.dot(weights) + bias
			weights_deriv = 2 * self.x_train.T.dot(y_predictions - self.y_train)
			bias_deriv = 2 * (y_predictions - self.y_train).sum()
			return weights_deriv, bias_deriv

		self.x_train = x_train
		self.y_train = y_train

		# Initial guesses and error
		weights_current = np.zeros(self.x_train.shape[1])
		bias_current = 0
		e_current = self.cost(self.x_train, self.y_train, weights_current, bias_current)
		self.cost_history.append(e_current)

		while True:
			weight_deriv, bias_deriv = calculate_gradients(weights_current, bias_current)

			weights_new = weights_current - weight_deriv * learning_rate
			bias_new = bias_current - bias_deriv * learning_rate
			e_new = self.cost(self.x_train, self.y_train, weights_new, bias_new)
			self.cost_history.append(e_new)

			# Stop if converged
			if abs(e_new - e_current) < converge_threshold:
				break

			# Decrease step size if error increases
			if e_new > e_current:
				learning_rate *= 0.9

			# Take the step
			weights_current, bias_current, e_current = weights_new, bias_new, e_new

		self.weights = weights_current
		self.bias = bias_current

	def cost(self, x, y, weights, bias):
		"""Mean absolute error"""
		y_predictions = x.dot(weights) + bias
		return np.abs(y_predictions - y).sum() / len(y)

	def predict(self, inputs):
		return inputs.dot(self.weights) + self.bias
