"""
Classifier class

Author: Sam Barba
Created 21/22/2021
"""

import numpy as np


EPSILON = 1e-9


class NaiveBayesClassifier:
	def __init__(self):
		self.labels = None
		self.means = None
		self.variances = None
		self.priors = None

	def fit(self, x, y):
		num_samples, num_features = x.shape
		labels, counts = np.unique(y, return_counts=True)
		self.labels = labels
		num_classes = sum(counts)

		# Calculate mean, variance, and prior for each class
		self.means = np.zeros((num_classes, num_features))
		self.variances = np.zeros((num_classes, num_features))
		self.priors = np.zeros(num_classes)

		for idx, c in enumerate(self.labels):
			xc = x[y == c]
			self.means[idx, :] = xc.mean(axis=0)
			self.variances[idx, :] = xc.var(axis=0)
			self.priors[idx] = len(xc) / num_samples

	def predict(self, x):
		probs = []

		for class_idx in self.labels:
			class_mean = self.means[class_idx]
			class_var = self.variances[class_idx] + EPSILON
			class_prior = self.priors[class_idx]
			class_likelihood = np.exp(-(x - class_mean) ** 2 / (2 * class_var))
			class_probs = np.prod(class_likelihood, axis=1)
			probs.append(class_prior * class_probs)

		probs = np.array(probs).T
		probs /= (probs.sum(axis=1, keepdims=True) + EPSILON)

		return probs
