"""
Perceptron classifier for perceptron.py

Author: Sam Barba
Created 23/11/2021
"""

import numpy as np

class PerceptronClf:
	def __init__(self):
		self.x_train = None
		self.y_train = None
		self.weights = None
		self.bias = 0

	def fit(self, x_train, y_train, max_iters=1000, learning_rate=0.01):
		self.x_train = x_train
		self.y_train = y_train
		self.weights = np.zeros(self.x_train.shape[1])

		for _ in range(max_iters):
			for idx, sample in enumerate(self.x_train):
				linear_model = sample.dot(self.weights) + self.bias
				y_pred = self.__unit_step(linear_model)

				update = learning_rate * (self.y_train[idx] - y_pred)

				self.weights += update * sample
				self.bias += update

	def predict(self, inputs):
		linear_model = inputs.dot(self.weights) + self.bias
		return self.__unit_step(linear_model)

	# Activation function
	def __unit_step(self, inputs):
		return np.where(inputs >= 0, 1, 0)
