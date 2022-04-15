# PCA transformer for principalComponentAnalysis.py
# Author: Sam Barba
# Created 10/11/2021

import numpy as np

class PCA:
	def __init__(self, num_components=2):
		self.num_components = num_components

	def transform(self, x):
		mean = np.mean(x, axis=0)
		x -= mean

		covariance = np.cov(x.T)
		variability = np.trace(covariance)

		eigenvalues, eigenvectors = np.linalg.eig(covariance)
		eigenvectors = eigenvectors.T
		indices = np.argsort(eigenvalues)[::-1]
		eigenvalues = eigenvalues[indices]
		eigenvectors = eigenvectors[indices]

		components = eigenvectors[:self.num_components]

		pca_variability = (eigenvalues / variability)[:self.num_components].sum()

		return np.dot(x, components.T), pca_variability
