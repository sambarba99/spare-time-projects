"""
Classifier for naiveBayes.py

Author: Sam Barba
Created 21/22/2021
"""

import numpy as np

class NaiveBayesClassifier:
	def __init__(self):
		self.x_train = None
		self.y_train = None
		self.classes = None
		self.means = None
		self.variances = None
		self.priors = None

	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	def train(self):
		num_samples, num_features = self.x_train.shape
		self.classes = np.unique(self.y_train)
		num_classes = len(self.classes)

		# Calculate mean, variance, and prior for each class
		self.means = np.zeros((num_classes, num_features))
		self.variances = np.zeros((num_classes, num_features))
		self.priors = np.zeros(num_classes)

		for idx, c in enumerate(self.classes):
			xc = self.x_train[self.y_train == c]
			self.means[idx, :] = np.mean(xc, axis=0)
			self.variances[idx, :] = np.var(xc, axis=0)
			self.priors[idx] = len(xc) / num_samples

	def predict(self, inputs):
		posteriors = []
		epsilon = 10 ** -6  # To avoid log errors

		# Calculate posterior probability for each class
		for class_idx, c in enumerate(self.classes):
			prior = np.log(self.priors[class_idx])
			posterior = np.log(self.__pdf(class_idx, inputs) + epsilon).sum() + prior
			posteriors.append(posterior)

		# Return class with the highest posterior probability
		return self.classes[np.argmax(posteriors)]

	def __pdf(self, class_idx, sample):
		mean = self.means[class_idx]
		var = self.variances[class_idx]
		num = np.exp(-((sample - mean) ** 2) / (2 * var))
		denom = (2 * np.pi * var) ** 0.5
		return num / denom
