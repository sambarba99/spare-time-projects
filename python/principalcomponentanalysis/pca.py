"""
PCA transformer for principal_component_analysis.py

Author: Sam Barba
Created 10/11/2021
"""

import numpy as np

class PCA:
	def __init__(self, n_components=2):
		self.n_components = n_components

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

		components = eigenvectors[:self.n_components]

		pca_variability = (eigenvalues / variability)[:self.n_components].sum()

		return x.dot(components.T), pca_variability
