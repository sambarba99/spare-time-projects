# Logistic regressor for logisticRegression.py
# Author: Sam Barba
# Created 10/11/2021

import numpy as np

class LogisticRegressor:
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
	def train(self, learning_rate=0.001, converge_threshold=10 ** -6):
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
		linear_model = np.dot(self.x_train, weights) + bias
		probs = self.__sigmoid(linear_model)
		weight_deriv = np.dot(self.x_train.T, probs - self.y_train)
		bias_deriv = (probs - self.y_train).sum()
		return weight_deriv, bias_deriv

	def cost(self, x, y, weights, bias):
		epsilon = 10 ** -6  # To avoid log errors
		linear_model = np.dot(x, weights) + bias
		probs = self.__sigmoid(linear_model)
		return -(y * np.log(probs + epsilon) + (1 - y) * np.log(1 - probs + epsilon)).sum() / len(x)

	def predict(self, inputs):
		linear_model = np.dot(inputs, self.weights) + self.bias
		probs = self.__sigmoid(linear_model)
		class_predictions = [1 if i > 0.5 else 0 for i in probs]
		return np.array(class_predictions)

	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
