"""
Logistic regressor for logistic_regression.py

Author: Sam Barba
Created 10/11/2021
"""

import numpy as np

class LogisticRegressor:
	def __init__(self):
		self.x_train = None
		self.y_train = None
		self.weights = None
		self.bias = 0
		self.cost_history = []

	def fit(self, x_train, y_train, learning_rate=0.001, converge_threshold=1e-6):
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

		self.x_train = x_train
		self.y_train = y_train

		# Initial guesses and error
		weights_current = np.zeros(self.x_train.shape[1])
		bias_current = 0
		e_current = cost(self.x_train, self.y_train, weights_current, bias_current)
		self.cost_history.append(e_current)

		while True:
			weight_deriv, bias_deriv = calculate_gradients(weights_current, bias_current)

			weights_new = weights_current - weight_deriv * learning_rate
			bias_new = bias_current - bias_deriv * learning_rate
			e_new = cost(self.x_train, self.y_train, weights_new, bias_new)
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

	def predict(self, inputs):
		linear_model = inputs.dot(self.weights) + self.bias
		probs = self.__sigmoid(linear_model)
		class_predictions = np.where(probs > 0.5, 1, 0)
		return class_predictions

	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
