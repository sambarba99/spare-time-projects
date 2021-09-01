"""
Neural Network class for digit_recognition_neural_network.py

Author: Sam Barba
Created 20/10/2021
"""

import numpy as np

class NeuralNetwork:
	def __init__(self, n_input_layer_neurons=784, n_hidden_layer_neurons=400, n_output_layer_neurons=10):
		"""
		Parameters:
			n_input_layer_neurons: 784 inputs from 28x28 image
			n_hidden_layer_neurons: Arbitrary amount of 400 hidden layer neurons
			n_output_layer_neurons: 10 prediction possibilities (0-9)
		"""

		self.x_train = None
		self.y_train = None
		# Sample weights from normal distribution
		self.hidden_weights = np.random.randn(n_hidden_layer_neurons, n_input_layer_neurons)
		self.hidden_bias = np.zeros((n_hidden_layer_neurons, 1))
		self.output_weights = np.random.randn(n_output_layer_neurons, n_hidden_layer_neurons)
		self.output_bias = np.zeros((n_output_layer_neurons, 1))
		self.loss = []

	def fit(self, x_train, y_train, iterations=1000, learning_rate=0.1):
		def sigmoid_derivative(x):
			return x * (1 - x)

		def calculate_loss(predictions, actual):
			return 0.5 * (predictions - actual) ** 2

		self.x_train = x_train
		self.y_train = y_train

		for i in range(iterations):
			if i % int(iterations * 0.001) == 0:
				# Print every 0.1%
				progress = 100 * i / iterations
				print(f'Training {progress:.1f}% done')

			iteration_loss = []

			for idx, item in enumerate(self.x_train):
				# Make vertical
				input_vector = item.reshape(-1, 1)
				actual = self.y_train[idx].reshape(-1, 1)

				hidden_layer_in = self.hidden_weights.dot(input_vector) + self.hidden_bias
				hidden_layer_out = self.__sigmoid(hidden_layer_in)

				output_layer_in = self.output_weights.dot(hidden_layer_out) + self.output_bias
				output_layer_out = self.__sigmoid(output_layer_in)  # Prediction vector

				error = actual - output_layer_out
				delta_output_layer_out = error * sigmoid_derivative(output_layer_out)

				error_hidden = delta_output_layer_out.T.dot(self.output_weights)
				delta_hidden_layer = error_hidden.T * sigmoid_derivative(hidden_layer_out)

				self.output_weights += hidden_layer_out.dot(delta_output_layer_out.T).T * learning_rate
				self.output_bias += delta_output_layer_out.sum(axis=0, keepdims=True) * learning_rate

				self.hidden_weights += input_vector.dot(delta_hidden_layer.T).T * learning_rate
				self.hidden_bias += delta_hidden_layer.sum(axis=0, keepdims=True) * learning_rate

				iteration_loss.append(np.mean(calculate_loss(output_layer_out, actual)))

			self.loss.append(np.mean(iteration_loss))

	def predict(self, input_vector):
		"""
		Return prediction vector e.g. v = [0.123, 0.047, 0.310, 0.968, 0.032, 0.045, 0.078, 0.123, 0.145, 0.227]
		In this case, np.argmax(v) = 3, therefore prediction is digit '3'
		"""

		# Make vertical
		input_vector = input_vector.reshape(-1, 1)

		hidden_layer_in = self.hidden_weights.dot(input_vector) + self.hidden_bias
		hidden_layer_out = self.__sigmoid(hidden_layer_in)
		output_layer_in = self.output_weights.dot(hidden_layer_out) + self.output_bias

		# Make horizontal again
		return self.__sigmoid(output_layer_in).reshape(1, -1)[0]

	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
