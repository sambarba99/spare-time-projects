"""
Principal Component Analysis demo

Author: Sam Barba
Created 10/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from _utils.csv_data_loader import load_csv_classification_data


plt.rcParams['figure.figsize'] = (7, 7)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)


def transform(x, num_components):
	x -= x.mean(axis=0)

	covariance = np.cov(x.T)
	eigenvalues, eigenvectors = np.linalg.eig(covariance)
	if isinstance(eigenvalues[0], np.complex128):
		eigenvalues = eigenvalues.real

	indices = eigenvalues.argsort()[::-1]
	eigenvalues = eigenvalues[indices]
	eigenvectors = eigenvectors.T[indices]

	components = eigenvectors[:num_components]
	x_transform = x.dot(components.T)

	variability = covariance.trace()
	total_explained_variance = (eigenvalues / variability)[:num_components].sum()

	return x_transform, total_explained_variance


if __name__ == '__main__':
	choice = input(
		'\nEnter 1 for banknote dataset,'
		'\n2 for breast tumour dataset,'
		'\n3 for glass dataset,'
		'\n4 for iris dataset,'
		'\n5 for mushroom dataset,'
		'\n6 for pulsar dataset,'
		'\n7 for Titanic dataset,'
		'\nor 8 for wine dataset\n>>> '
	)

	match choice:
		case '1': path = 'C:/Users/Sam/Desktop/projects/datasets/banknote_authenticity.csv'
		case '2': path = 'C:/Users/Sam/Desktop/projects/datasets/breast_tumour_pathology.csv'
		case '3': path = 'C:/Users/Sam/Desktop/projects/datasets/glass_classification.csv'
		case '4': path = 'C:/Users/Sam/Desktop/projects/datasets/iris_classification.csv'
		case '5': path = 'C:/Users/Sam/Desktop/projects/datasets/mushroom_edibility_classification.csv'
		case '6': path = 'C:/Users/Sam/Desktop/projects/datasets/pulsar_identification.csv'
		case '7': path = 'C:/Users/Sam/Desktop/projects/datasets/titanic_survivals.csv'
		case _: path = 'C:/Users/Sam/Desktop/projects/datasets/wine_classification.csv'

	num_components = None
	while num_components not in ('2', '3'):
		num_components = input('\nEnter no. components (2 or 3)\n>>> ')
	num_components = int(num_components)

	# Normalise x (PCA is sensitive to feature scales)
	x, y, labels, _ = load_csv_classification_data(path, x_transform=MinMaxScaler())
	x_transform, explained_variance_ratio = transform(x, num_components)

	ax = plt.axes() if num_components == 2 else plt.axes(projection='3d')
	scatter = ax.scatter(*x_transform.T, c=y, alpha=0.5, cmap='brg') \
		if num_components == 2 else \
		ax.scatter3D(*x_transform.T, c=y, alpha=0.5, cmap='brg')
	ax.set_xlabel('Principal component 1')
	ax.set_ylabel('Principal component 2')
	if num_components == 3:
		x_plt, y_plt, z_plt = x_transform.T
		ax.plot(y_plt, z_plt, 'k.', markersize=2, alpha=0.4, zdir='x', zs=x_plt.min() - 0.1)
		ax.plot(x_plt, z_plt, 'k.', markersize=2, alpha=0.4, zdir='y', zs=y_plt.max() + 0.1)
		ax.plot(x_plt, y_plt, 'k.', markersize=2, alpha=0.4, zdir='z', zs=z_plt.min() - 0.1)
		ax.set_zlabel('Principal component 3')
	ax.set_title(
		fr'Shape of $x$: {x.shape}'
		f'\nShape of PCA transform: {x_transform.shape}'
		f'\nExplained variance ratio: {explained_variance_ratio:.4f}'
	)
	handles, _ = scatter.legend_elements()
	for h in handles:
		h.set_alpha(1)
	ax.legend(handles, labels)
	plt.show()
