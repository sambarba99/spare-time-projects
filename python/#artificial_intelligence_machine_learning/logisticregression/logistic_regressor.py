"""
Logistic regressor class

Author: Sam Barba
Created 10/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np


EPSILON = 1e-9


class LogisticRegressor:
	def __init__(self, labels):
		self.weights = None
		self.bias = 0
		self.cost_history = []
		self.labels = labels


	def fit(self, x, y, learning_rate=1e-4, converge_threshold=2e-4):
		"""Gradient descent solution"""

		def cost(x, y, weights, bias):
			linear_model = x.dot(weights) + bias
			probs = self.__sigmoid(linear_model)

			return -(y * np.log(probs + EPSILON) + (1 - y) * np.log(1 - probs + EPSILON)).sum() / len(x)


		def calculate_gradients(weights, bias):
			linear_model = x.dot(weights) + bias
			probs = self.__sigmoid(linear_model)
			weight_deriv = x.T.dot(probs - y)
			bias_deriv = (probs - y).sum()

			return weight_deriv, bias_deriv


		def plot_decision_boundary(weights, bias, converged, first_time):
			w1, w2 = weights
			m = -w1 / w2 if not first_time else None
			c = -bias / w2 if not first_time else None

			# Set up boundaries
			x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
			y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5

			plt.cla()
			for idx, label in enumerate(self.labels):
				plt.scatter(*x[y == idx].T, alpha=0.5, label=label)
			if not first_time:
				line_x = np.array([x_min, x_max])
				line_y = m * line_x + c
				# plt.plot(line_x, line_y, color='black', linewidth=1, ls='--')
				plt.fill_between(line_x, line_y, -100, color='tab:orange', alpha=0.2)
				plt.fill_between(line_x, line_y, 100, color='tab:blue', alpha=0.2)
			plt.xlim(x_min, x_max)
			plt.ylim(y_min, y_max)
			plt.xlabel(r'$x_1$ (Principal component 1)')
			plt.ylabel(r'$x_2$ (Principal component 2)')
			plt.title(
				'Start' if first_time else
				fr'Gradient descent solution: $m$ = {m:.3f}  |  $c$ = {c:.3f}' + f'\n(converged: {converged})'
			)
			legend = plt.legend()
			for handle in legend.legend_handles:
				handle.set_alpha(1)

			if converged:
				plt.show()
			else:
				plt.draw()
				plt.pause(2 if first_time else 1e-6)


		# Initial guesses and error (arbitrary)
		weights_current = np.array([1, 0])
		bias_current = 0
		e_current = cost(x, y, weights_current, bias_current)
		self.cost_history.append(e_current)
		plot_decision_boundary(weights_current, bias_current, converged=False, first_time=True)

		i = 0
		while True:
			i += 1

			weight_deriv, bias_deriv = calculate_gradients(weights_current, bias_current)

			weights_new = weights_current - weight_deriv * learning_rate
			bias_new = bias_current - bias_deriv * learning_rate
			e_new = cost(x, y, weights_new, bias_new)
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


	def predict(self, x):
		linear_model = x.dot(self.weights) + self.bias
		probs = self.__sigmoid(linear_model)

		return probs


	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
