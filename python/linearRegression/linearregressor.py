# Linear regressor for linearRegression.py
# Author: Sam Barba
# Created 10/11/2021

import numpy as np

class LinearRegressor:
	def __init__(self):
		self.x_train = None
		self.y_train = None
		self.weights = None
		self.bias = 0
		self.cost_history = []

	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	# Gradient descent
	def train(self, learning_rate=0.0001, converge_threshold=10 ** -9):
		# Initial guesses and error
		weights_current = np.zeros(self.x_train.shape[1])
		bias_current = 0
		e_current = self.cost(self.x_train, self.y_train, weights_current, bias_current)
		self.cost_history.append(e_current)

		while True:
			weight_deriv, bias_deriv = self.__calculate_gradients(weights_current, bias_current)

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

	def __calculate_gradients(self, weights, bias):
		y_predictions = np.dot(self.x_train, weights) + bias
		weights_deriv = 2 * np.dot(self.x_train.T, y_predictions - self.y_train)
		bias_deriv = 2 * (y_predictions - self.y_train).sum()
		return weights_deriv, bias_deriv

	# Least squares error
	def cost(self, x, y, weights, bias):
		y_predictions = np.dot(x, weights) + bias
		return ((y - y_predictions) ** 2).sum()

	def predict(self, inputs):
		return np.dot(inputs, self.weights) + self.bias
