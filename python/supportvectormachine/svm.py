"""
SVM classifier for support_vector_machine.py

Author: Sam Barba
Created 22/11/2021
"""

import numpy as np

class SVM:
	def __init__(self):
		self.x_train = None
		self.y_train = None
		self.weights = None
		self.bias = 0

	def fit(self, x_train, y_train, max_iters=1000, lambda_param=0.01, learning_rate=0.001):
		self.x_train = x_train
		self.y_train = y_train
		self.weights = np.zeros(self.x_train.shape[1])

		for _ in range(max_iters):
			for idx, sample in enumerate(self.x_train):
				if self.y_train[idx] * (sample.dot(self.weights) - self.bias) >= 1:
					self.weights -= (2 * lambda_param * self.weights) * learning_rate
				else:
					self.weights -= (2 * lambda_param * self.weights - sample.dot(self.y_train[idx])) * learning_rate
					self.bias -= self.y_train[idx] * learning_rate

	def predict(self, inputs):
		linear_model = inputs.dot(self.weights) - self.bias
		return np.sign(linear_model).astype(int)
