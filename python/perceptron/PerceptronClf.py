# Perceptron classifier for perceptron.py
# Author: Sam Barba
# Created 23/11/2021

import numpy as np

class PerceptronClf:
	def __init__(self):
		self.xTrain = None
		self.yTrain = None
		self.weights = None
		self.bias = 0

	def fit(self, xTrain, yTrain):
		self.xTrain = xTrain
		self.yTrain = yTrain

	def train(self, maxIters=1000, learningRate=0.01):
		self.weights = np.zeros(self.xTrain.shape[1])

		for _ in range(maxIters):
			for idx, sample in enumerate(self.xTrain):
				linearModel = np.dot(sample, self.weights) + self.bias
				yPred = self.__unitStep(linearModel)

				update = learningRate * (self.yTrain[idx] - yPred)

				self.weights += update * sample
				self.bias += update

	def predict(self, inputs):
		linearModel = np.dot(inputs, self.weights) + self.bias
		return self.__unitStep(linearModel)

	# Activation function
	def __unitStep(self, inputs):
		return np.where(inputs >= 0, 1, 0)
