# Linear regressor for linearRegression.py
# Author: Sam Barba
# Created 10/11/2021

import numpy as np

class LinearRegressor:
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
	def train(self, learningRate=0.0001, convergeThreshold=10 ** -9):
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
		yPredictions = np.dot(self.xTrain, weights) + bias
		weightsDeriv = 2 * np.dot(self.xTrain.T, yPredictions - self.yTrain)
		biasDeriv = 2 * (yPredictions - self.yTrain).sum()
		return weightsDeriv, biasDeriv

	# Least squares error
	def cost(self, x, y, weights, bias):
		yPredictions = np.dot(x, weights) + bias
		return ((y - yPredictions) ** 2).sum()

	def predict(self, inputs):
		return np.dot(inputs, self.weights) + self.bias
