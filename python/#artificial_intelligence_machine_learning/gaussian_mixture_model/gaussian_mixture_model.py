"""
Gaussian Mixture Model class

Author: Sam Barba
Created 15/12/2023
"""

import numpy as np
from scipy.stats import multivariate_normal


EPSILON = 1e-9


class GaussianMixtureModel:
	def __init__(self, n_components):
		self.n_components = n_components
		self.weights = None
		self.means = None
		self.covariances = None

	def compute_expectation(self, x):
		likelihoods = np.array([
			multivariate_normal.pdf(x, mean=mean, cov=cov, allow_singular=True)
			for mean, cov in zip(self.means, self.covariances)
		]).T
		weighted_likelihoods = likelihoods * self.weights + EPSILON
		responsibilities = weighted_likelihoods / weighted_likelihoods.sum(axis=1, keepdims=True)

		return responsibilities

	def compute_maximisation(self, x, responsibilities):
		total_responsibilities = responsibilities.sum(axis=0)
		self.weights = total_responsibilities / len(x)
		self.means = responsibilities.T.dot(x) / total_responsibilities[:, np.newaxis]
		self.covariances = [
			(x - mean).T.dot((x - mean) * responsibility[:, np.newaxis]) / total_resp
			for mean, responsibility, total_resp in zip(self.means, responsibilities.T, total_responsibilities)
		]

	def fit(self, x, max_iters=1000, tolerance=1e-6):
		# Initialise params
		self.weights = np.ones(self.n_components) / self.n_components
		prev_weights = self.weights.copy()
		self.means = x[np.random.choice(len(x), self.n_components, replace=False)]
		self.covariances = np.array([np.cov(x.T) for _ in range(self.n_components)])

		for i in range(1, max_iters + 1):
			responsibilities = self.compute_expectation(x)
			self.compute_maximisation(x, responsibilities)

			# Check for convergence
			if i > 0 and np.abs(prev_weights - self.weights).max() < tolerance:
				print(f'Stopping at iter {i}/{max_iters}')
				break

			prev_weights = self.weights.copy()

	def predict(self, X):
		responsibilities = self.compute_expectation(X)
		return np.argmax(responsibilities, axis=1)
