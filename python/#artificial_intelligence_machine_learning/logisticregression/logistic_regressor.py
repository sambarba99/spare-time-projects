"""
Logistic regressor class

Author: Sam Barba
Created 10/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np

class LogisticRegressor:
	def __init__(self, labels):
		self.x_train = None
		self.y_train = None
		self.weights = None
		self.bias = 0
		self.cost_history = []
		self.labels = labels

	def fit(self, x_train, y_train, learning_rate=1e-4, converge_threshold=1e-4):
		"""Gradient descent"""

		def cost(x, y, weights, bias):
			epsilon = 1e-6  # To avoid log errors
			linear_model = x.dot(weights) + bias
			probs = self.__sigmoid(linear_model)
			return -(y * np.log(probs + epsilon) + (1 - y) * np.log(1 - probs + epsilon)).sum() / len(x)

		def calculate_gradients(weights, bias):
			linear_model = self.x_train.dot(weights) + bias
			probs = self.__sigmoid(linear_model)
			weight_deriv = self.x_train.T.dot(probs - self.y_train)
			bias_deriv = (probs - self.y_train).sum()
			return weight_deriv, bias_deriv

		def plot_decision_boundary(weights, bias, converged, first_time):
			"""
			See https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/
			for calculation of m and c
			"""

			w1, w2 = weights
			m = -w1 / w2 if not first_time else None
			c = -bias / w2 if not first_time else None

			# Set up boundaries
			x_min, x_max = self.x_train[:, 0].min() - 0.5, self.x_train[:, 0].max() + 0.5
			y_min, y_max = self.x_train[:, 1].min() - 0.5, self.x_train[:, 1].max() + 0.5

			plt.cla()
			for idx, label in enumerate(self.labels):
				plt.scatter(*self.x_train[self.y_train == idx].T, alpha=0.7, label=label)
			if not first_time:
				line_x = np.array([x_min, x_max])
				line_y = m * line_x + c
				# plt.plot(line_x, line_y, color='black', linewidth=1, ls='--')
				plt.fill_between(line_x, line_y, -100, color='tab:orange', alpha=0.4)
				plt.fill_between(line_x, line_y, 100, color='tab:blue', alpha=0.4)
			plt.xlim(x_min, x_max)
			plt.ylim(y_min, y_max)
			plt.xlabel(r'$x_1$ (Principal component 1)')
			plt.ylabel(r'$x_2$ (Principal component 2)')
			if first_time:
				plt.title('Start')
			else:
				plt.title(fr'Gradient descent solution: $m$ = {m:.3f}  |  $c$ = {c:.3f}' + f'\n(converged: {converged})')
			plt.legend()

			plt.show(block=converged)
			if not converged: plt.pause(2 if first_time else 1e-6)

		self.x_train = x_train
		self.y_train = y_train

		# Initial guesses and error (arbitrary)
		weights_current = np.array([1, 0])
		bias_current = 0
		e_current = cost(self.x_train, self.y_train, weights_current, bias_current)
		self.cost_history.append(e_current)
		plot_decision_boundary(weights_current, bias_current, converged=False, first_time=True)

		i = 0
		while True:
			i += 1

			weight_deriv, bias_deriv = calculate_gradients(weights_current, bias_current)

			weights_new = weights_current - weight_deriv * learning_rate
			bias_new = bias_current - bias_deriv * learning_rate
			e_new = cost(self.x_train, self.y_train, weights_new, bias_new)
			self.cost_history.append(e_new)

			# Stop if converged
			if abs(e_new - e_current) < converge_threshold:
				plot_decision_boundary(weights_new, bias_new, converged=True, first_time=False)
				break

			# Decrease step size if error increases
			if e_new > e_current:
				learning_rate *= 0.9

			# Take the step
			weights_current, bias_current, e_current = weights_new, bias_new, e_new

			if i % 2 == 0: plot_decision_boundary(weights_current, bias_current, converged=False, first_time=False)

		self.weights = weights_current
		self.bias = bias_current

	def predict(self, inputs):
		linear_model = inputs.dot(self.weights) + self.bias
		probs = self.__sigmoid(linear_model)
		class_predictions = probs.round().astype(int)
		return class_predictions

	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
