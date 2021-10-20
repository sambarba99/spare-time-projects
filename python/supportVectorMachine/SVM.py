# SVM classifier for supportVectorMachine.py
# Author: Sam Barba
# Created 22/11/2021

import numpy as np

class SVM:
	def __init__(self):
		self.xTrain = None
		self.yTrain = None
		self.weights = None
		self.bias = 0

	def fit(self, xTrain, yTrain):
		self.xTrain = xTrain
		self.yTrain = yTrain

	def train(self, maxIters=1000, lambdaParam=0.01, learningRate=0.001):
		self.weights = np.zeros(self.xTrain.shape[1])

		for i in range(maxIters):
			for idx, sample in enumerate(self.xTrain):
				if self.yTrain[idx] * (np.dot(sample, self.weights) - self.bias) >= 1:
					self.weights -= (2 * lambdaParam * self.weights) * learningRate
				else:
					self.weights -= (2 * lambdaParam * self.weights - np.dot(sample, self.yTrain[idx])) * learningRate
					self.bias -= self.yTrain[idx] * learningRate

	def predict(self, inputs):
		linearModel = np.dot(inputs, self.weights) - self.bias
		return np.sign(linearModel).astype(int)
