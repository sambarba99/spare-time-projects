"""
Linear Discriminant Analysis demo

Author: Sam Barba
Created 12/03/2024
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


def transform(x, y, num_components):
	num_features = x.shape[1]
	mean_overall = np.mean(x, axis=0)
	sw = np.zeros((num_features, num_features))
	sb = np.zeros((num_features, num_features))

	for class_lbl in np.unique(y):
		x_c = x[y == class_lbl]
		mean_c = np.mean(x_c, axis=0)
		sw += (x_c - mean_c).T.dot((x_c - mean_c))

		n_c = x_c.shape[0]
		mean_diff = (mean_c - mean_overall).reshape(num_features, 1)
		sb += n_c * mean_diff.dot(mean_diff.T)

	# Determine SW^-1 * SB
	a = np.linalg.inv(sw).dot(sb)

	# Get eigenvalues and eigenvectors of SW^-1 * SB
	eigenvalues, eigenvectors = np.linalg.eig(a)
	if isinstance(eigenvalues[0], np.complex128):
		eigenvalues = eigenvalues.real

	indices = np.abs(eigenvalues).argsort()[::-1]
	eigenvectors = eigenvectors.T[indices]

	components = eigenvectors[:num_components]
	x_transform = np.dot(x, components.T)

	return x_transform


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

	# Normalise x (LDA is sensitive to feature scales)
	x, y, labels, _ = load_csv_classification_data(path, x_transform=MinMaxScaler())
	x_transform = transform(x, y, num_components)

	ax = plt.axes() if num_components == 2 else plt.axes(projection='3d')
	scatter = ax.scatter(*x_transform.T, c=y, alpha=0.5, cmap='brg') \
		if num_components == 2 else \
		ax.scatter3D(*x_transform.T, c=y, alpha=0.5, cmap='brg')
	ax.set_xlabel('Linear discriminant 1')
	ax.set_ylabel('Linear discriminant 2')
	if num_components == 3:
		x_plt, y_plt, z_plt = x_transform.T
		ax.plot(y_plt, z_plt, 'k.', markersize=2, alpha=0.4, zdir='x', zs=x_plt.min() - 0.1)
		ax.plot(x_plt, z_plt, 'k.', markersize=2, alpha=0.4, zdir='y', zs=y_plt.max() + 0.1)
		ax.plot(x_plt, y_plt, 'k.', markersize=2, alpha=0.4, zdir='z', zs=z_plt.min() - 0.1)
		ax.set_zlabel('Linear discriminant 3')
	ax.set_title(
		fr'Shape of $x$: {x.shape}'
		f'\nShape of LDA transform: {x_transform.shape}'
	)
	handles, _ = scatter.legend_elements()
	for h in handles:
		h.set_alpha(1)
	ax.legend(handles, labels)
	plt.show()
