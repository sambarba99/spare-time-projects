"""
Linear regressor for linear_regression.py

Author: Sam Barba
Created 10/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np

class LinearRegressor:
	def __init__(self, feature, y_name):
		self.x_train = None
		self.y_train = None
		self.weights = None
		self.bias = 0
		self.cost_history = []
		self.feature = feature
		self.y_name = y_name

	def fit(self, x_train, y_train, learning_rate=1e-5, converge_threshold=1e-5):
		"""Gradient descent"""

		def calculate_gradients(weights, bias):
			y_predictions = self.x_train.dot(weights) + bias
			weights_deriv = 2 * self.x_train.T.dot(y_predictions - self.y_train)
			bias_deriv = 2 * (y_predictions - self.y_train).sum()
			return weights_deriv, bias_deriv

		def plot_regression_line(weights, bias, converged, first_time):
			padding = 2
			line_x = np.array([np.min(self.x_train) - padding, np.max(self.x_train) + padding]) \
				if not first_time else None
			line_y = weights[0][0] * line_x + bias \
				if not first_time else None

			plt.cla()
			plt.scatter(self.x_train, self.y_train, alpha=0.7)
			if not first_time:
				plt.plot(line_x, line_y, color='black', linewidth=1)
			padding = 0.5
			plt.xlim(np.min(self.x_train) - padding, np.max(self.x_train) + padding)
			plt.ylim(np.min(self.y_train) - padding, np.max(self.y_train) + padding)
			plt.xlabel(fr'$x$ ({self.feature}) (standardised)')
			plt.ylabel(fr'$y$ ({self.y_name})')
			if first_time:
				plt.title('Start')
			else:
				plt.title(fr'Gradient descent solution: $m$ = {weights[0][0]:.3f}  |  $c$ = {bias:.3f}'
					+ f'\n(converged: {converged})')

			plt.show(block=converged)
			if not converged: plt.pause(2 if first_time else 1e-6)

		self.x_train = x_train
		self.y_train = y_train

		# Initial guesses and error
		weights_current = np.zeros(self.x_train.shape[1])
		bias_current = 0
		e_current = self.cost(self.x_train, self.y_train, weights_current, bias_current)
		self.cost_history.append(e_current)
		plot_regression_line(weights_current, bias_current, converged=False, first_time=True)

		i = 0
		while True:
			i += 1

			weight_deriv, bias_deriv = calculate_gradients(weights_current, bias_current)

			weights_new = weights_current - weight_deriv * learning_rate
			bias_new = bias_current - bias_deriv * learning_rate
			e_new = self.cost(self.x_train, self.y_train, weights_new, bias_new)
			self.cost_history.append(e_new)

			# Stop if converged
			if abs(e_new - e_current) < converge_threshold:
				plot_regression_line(weights_new, bias_new, converged=True, first_time=False)
				break

			# Decrease step size if error increases
			if e_new > e_current:
				learning_rate *= 0.1

			# Take the step
			weights_current, bias_current, e_current = weights_new, bias_new, e_new

			if i % 25 == 0: plot_regression_line(weights_current, bias_current, converged=False, first_time=False)

		self.weights = weights_current
		self.bias = bias_current

	def cost(self, x, y, weights, bias):
		"""Mean absolute error"""

		y_predictions = x.dot(weights) + bias
		return np.abs(y_predictions - y).sum() / len(y)

	def predict(self, inputs):
		return inputs.dot(self.weights) + self.bias
