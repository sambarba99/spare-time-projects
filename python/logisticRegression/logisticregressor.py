# Logistic regressor for logisticRegression.py
# Author: Sam Barba
# Created 10/11/2021

import numpy as np

class LogisticRegressor:
	def __init__(self):
		self.xTrain = None
		self.yTrain = None
		self.weights = None
		self.bias = 0
		self.costHistory = []

	def fit(self, xTrain, yTrain):
		self.xTrain = xTrain
		self.yTrain = yTrain

	# Gradient descent
	def train(self, learningRate=0.001, convergeThreshold=10 ** -6):
		# Initial guesses and error
		weightsCurrent = np.zeros(self.xTrain.shape[1])
		biasCurrent = 0
		eCurrent = self.cost(self.xTrain, self.yTrain, weightsCurrent, biasCurrent)
		self.costHistory.append(eCurrent)

		while True:
			weightDeriv, biasDeriv = self.__calculateGradients(weightsCurrent, biasCurrent)

			weightsNew = weightsCurrent - weightDeriv * learningRate
			biasNew = biasCurrent - biasDeriv * learningRate
			eNew = self.cost(self.xTrain, self.yTrain, weightsNew, biasNew)
			self.costHistory.append(eNew)

			# Stop if converged
			if abs(eNew - eCurrent) < convergeThreshold:
				break

			# Decrease step size if error increases
			if eNew > eCurrent:
				learningRate *= 0.9

			# Take the step
			weightsCurrent, biasCurrent, eCurrent = weightsNew, biasNew, eNew

		self.weights = weightsCurrent
		self.bias = biasCurrent

	def __calculateGradients(self, weights, bias):
		linearModel = np.dot(self.xTrain, weights) + bias
		probs = self.__sigmoid(linearModel)
		weightDeriv = np.dot(self.xTrain.T, probs - self.yTrain)
		biasDeriv = (probs - self.yTrain).sum()
		return weightDeriv, biasDeriv

	def cost(self, x, y, weights, bias):
		epsilon = 10 ** -6 # To avoid log errors
		linearModel = np.dot(x, weights) + bias
		probs = self.__sigmoid(linearModel)
		return -(y * np.log(probs + epsilon) + (1 - y) * np.log(1 - probs + epsilon)).sum() / len(x)

	def predict(self, inputs):
		linearModel = np.dot(inputs, self.weights) + self.bias
		probs = self.__sigmoid(linearModel)
		classPredictions = [1 if i > 0.5 else 0 for i in probs]
		return np.array(classPredictions)

	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
