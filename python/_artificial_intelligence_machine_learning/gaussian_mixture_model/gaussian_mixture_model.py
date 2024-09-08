"""
Gaussian Mixture Model class

Author: Sam Barba
Created 15/12/2023
"""

import numpy as np
from scipy.stats import multivariate_normal


EPSILON = 1e-9


class GaussianMixtureModel:
	def __init__(self, num_components):
		self.num_components = num_components
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

	def compute_log_likelihood(self, x):
		likelihoods = np.array([
			multivariate_normal.pdf(x, mean=mean, cov=cov, allow_singular=True)
			for mean, cov in zip(self.means, self.covariances)
		]).T

		return np.log(np.sum(likelihoods * self.weights, axis=1) + EPSILON).sum()

	def fit(self, x, max_iters=1000):
		# Initialise params
		self.weights = np.ones(self.num_components) / self.num_components
		self.means = x[np.random.choice(len(x), self.num_components, replace=False)]
		self.covariances = np.array([np.cov(x.T) for _ in range(self.num_components)])

		prev_log_likelihood = -np.inf

		for i in range(max_iters):
			responsibilities = self.compute_expectation(x)
			self.compute_maximisation(x, responsibilities)
			log_likelihood = self.compute_log_likelihood(x)

			# Check for convergence
			if abs(log_likelihood - prev_log_likelihood) < EPSILON:
				print(f'Stopping at iter {i + 1}/{max_iters}')
				break

			prev_log_likelihood = log_likelihood

	def predict(self, x):
		probs = self.compute_expectation(x)
		return probs
