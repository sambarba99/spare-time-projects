"""
PCA transformer for principal_component_analysis.py

Author: Sam Barba
Created 10/11/2021
"""

import numpy as np

def transform(x, n_components=2):
	x -= np.mean(x, axis=0)

	covariance = np.cov(x.T)
	variability = np.trace(covariance)

	eigenvalues, eigenvectors = np.linalg.eig(covariance)
	eigenvectors = eigenvectors.T
	indices = np.argsort(eigenvalues)[::-1]
	eigenvalues = eigenvalues[indices]
	eigenvectors = eigenvectors[indices]

	components = eigenvectors[:n_components]

	pca_variability = (eigenvalues / variability)[:n_components].sum()

	return x.dot(components.T), pca_variability
