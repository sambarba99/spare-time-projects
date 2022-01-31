# Classifier for naiveBayes.py
# Author: Sam Barba
# Created 21/22/2021

import numpy as np

class NaiveBayesClassifier:
	def __init__(self):
		self.xTrain = None
		self.yTrain = None
		self.classes = None
		self.means = None
		self.variances = None
		self.priors = None

	def fit(self, xTrain, yTrain):
		self.xTrain = xTrain
		self.yTrain = yTrain

	def train(self):
		numSamples, numFeatures = self.xTrain.shape
		self.classes = np.unique(self.yTrain)
		numClasses = len(self.classes)

		# Calculate mean, variance, and prior for each class
		self.means = np.zeros((numClasses, numFeatures))
		self.variances = np.zeros((numClasses, numFeatures))
		self.priors = np.zeros(numClasses)

		for idx, c in enumerate(self.classes):
			xc = self.xTrain[self.yTrain == c]
			self.means[idx, :] = np.mean(xc, axis=0)
			self.variances[idx, :] = np.var(xc, axis=0)
			self.priors[idx] = len(xc) / numSamples

	def predict(self, inputs):
		posteriors = []
		epsilon = 10 ** -6 # To avoid log errors

		# Calculate posterior probability for each class
		for classIdx, c in enumerate(self.classes):
			prior = np.log(self.priors[classIdx])
			posterior = np.log(self.__pdf(classIdx, inputs) + epsilon).sum() + prior
			posteriors.append(posterior)

		# Return class with the highest posterior probability
		return self.classes[np.argmax(posteriors)]

	def __pdf(self, classIdx, sample):
		mean = self.means[classIdx]
		var = self.variances[classIdx]
		num = np.exp(-((sample - mean) ** 2) / (2 * var))
		denom = (2 * np.pi * var) ** 0.5
		return num / denom
