# Neural Network class for digitRecognitionNeuralNetwork.py
# Author: Sam Barba
# Created 20/10/2021

import numpy as np

class NeuralNetwork:
	# 784 input layer neurons (784 inputs from 28*28 image)
	# Arbitrary amount of 50 hidden layer neurons
	# 10 output layer neurons (10 prediction possibilities, 0-9)
	def __init__(self, numInputLayerNeurons=784, numHiddenLayerNeurons=50, numOutputLayerNeurons=10):
		self.xTrain = None
		self.yTrain = None
		# Sample weights from normal distribution
		self.hiddenWeights = np.random.randn(numHiddenLayerNeurons, numInputLayerNeurons)
		self.hiddenBias = np.zeros((numHiddenLayerNeurons, 1))
		self.outputWeights = np.random.randn(numOutputLayerNeurons, numHiddenLayerNeurons)
		self.outputBias = np.zeros((numOutputLayerNeurons, 1))
		self.loss = []

	def fit(self, xTrain, yTrain):
		self.xTrain = xTrain
		self.yTrain = yTrain

	def train(self, iterations=1000, learningRate=0.1):
		for t in range(iterations):
			if t % int(iterations * 0.05) == 0:
				print("Training {}% done".format(round(100 * t / iterations, 1)))

			iterationLoss = []

			for idx, item in enumerate(self.xTrain):
				# Make vertical
				inputVector = item.reshape(-1,1)
				actual = self.yTrain[idx].reshape(-1,1)

				hiddenLayerIn = np.dot(self.hiddenWeights, inputVector) + self.hiddenBias
				hiddenLayerOut = self.__sigmoid(hiddenLayerIn)

				outputLayerIn = np.dot(self.outputWeights, hiddenLayerOut) + self.outputBias
				outputLayerOut = self.__sigmoid(outputLayerIn) # Prediction vector

				error = actual - outputLayerOut
				deltaOutputLayerOut = error * self.__sigmoidDerivative(outputLayerOut)

				errorHidden = np.dot(deltaOutputLayerOut.T, self.outputWeights)
				deltaHiddenLayer = errorHidden.T * self.__sigmoidDerivative(hiddenLayerOut)

				self.outputWeights += np.dot(hiddenLayerOut, deltaOutputLayerOut.T).T * learningRate
				self.outputBias += deltaOutputLayerOut.sum(axis=0, keepdims=True) * learningRate

				self.hiddenWeights += np.dot(inputVector, deltaHiddenLayer.T).T * learningRate
				self.hiddenBias += deltaHiddenLayer.sum(axis=0, keepdims=True) * learningRate

				iterationLoss.append(np.average(self.__calculateLoss(outputLayerOut, actual)))

			self.loss.append(np.average(iterationLoss))

	# Return prediction vector e.g. [0.123, 0.047, 0.310, 0.968, 0.032, 0.045, 0.078, 0.123, 0.145, 0.227]
	# np.argmax of this = 3, therefore prediction is digit '3'
	def predict(self, inputVector):
		# Make vertical
		inputVector = inputVector.reshape(-1,1)

		hiddenLayerIn = np.dot(self.hiddenWeights, inputVector) + self.hiddenBias
		hiddenLayerOut = self.__sigmoid(hiddenLayerIn)
		outputLayerIn = np.dot(self.outputWeights, hiddenLayerOut) + self.outputBias

		# Make horizontal again
		return self.__sigmoid(outputLayerIn).reshape(1,-1)[0]

	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def __sigmoidDerivative(self, x):
		return x * (1 - x)

	def __calculateLoss(self, predictions, actual):
		return 0.5 * (predictions - actual) ** 2
