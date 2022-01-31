# PCA transformer for principalComponentAnalysis.py
# Author: Sam Barba
# Created 10/11/2021

import numpy as np

class PCA:
	def __init__(self, numComponents=2):
		self.numComponents = numComponents

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

		components = eigenvectors[:self.numComponents]

		pcaVariability = np.sum((eigenvalues / variability)[:self.numComponents])

		return np.dot(x, components.T), pcaVariability
