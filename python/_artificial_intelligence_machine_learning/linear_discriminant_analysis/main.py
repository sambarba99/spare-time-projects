"""
Linear Discriminant Analysis demo

Author: Sam Barba
Created 12/03/2024
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.preprocessing import MinMaxScaler

from _utils.csv_data_loader import load_csv_classification_data


plt.rcParams['figure.figsize'] = (7, 7)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)


def transform(x, y, num_components):
	num_features = x.shape[1]
	x_mean = np.mean(x, axis=0)
	sw = np.zeros((num_features, num_features))  # Within-class scatter
	sb = np.zeros((num_features, num_features))  # Between-class scatter

	for class_lbl in np.unique(y):
		x_class = x[y == class_lbl]
		class_mean = np.mean(x_class, axis=0)
		class_centered = x_class - class_mean
		sw += class_centered.T.dot(class_centered)

		num_class = x_class.shape[0]
		mean_diff = class_mean - x_mean
		sb += num_class * np.outer(mean_diff, mean_diff)

	# Find directions that minimise within-class variance and maximise between-class variance
	eigenvalues, eigenvectors = eigh(sb, sw)

	# Sort eigenvectors by importance (largest eigenvalues first)
	indices = eigenvalues.argsort()[::-1]
	eigenvectors = eigenvectors[:, indices]

	# Select top components
	components = eigenvectors[:, :num_components]

	# Project data into this lower-dimensional space
	x_centered = x - x_mean
	x_transform = x_centered.dot(components)

	return x_transform


if __name__ == '__main__':
	choice = input(
		'\nEnter 1 for banknote dataset,'
		'\n2 for breast tumour dataset,'
		'\n3 for glass dataset,'
		'\n4 for iris dataset,'
		'\n5 for pulsar dataset,'
		'\n6 for Titanic dataset,'
		'\nor 7 for wine dataset\n>>> '
	)

	match choice:
		case '1': path = 'C:/Users/sam/Desktop/projects/datasets/banknote_authenticity.csv'
		case '2': path = 'C:/Users/sam/Desktop/projects/datasets/breast_tumour_pathology.csv'
		case '3': path = 'C:/Users/sam/Desktop/projects/datasets/glass_classification.csv'
		case '4': path = 'C:/Users/sam/Desktop/projects/datasets/iris_classification.csv'
		case '5': path = 'C:/Users/sam/Desktop/projects/datasets/pulsar_identification.csv'
		case '6': path = 'C:/Users/sam/Desktop/projects/datasets/titanic_survivals.csv'
		case _: path = 'C:/Users/sam/Desktop/projects/datasets/wine_classification.csv'

	# Normalise x (LDA is sensitive to feature scales)
	x, y, labels, _ = load_csv_classification_data(path, x_transform=MinMaxScaler())
	cmap = 'bwr' if len(labels) == 2 else 'brg'

	for num_components in (2, 3):
		x_transform = transform(x, y, num_components)

		ax = plt.axes() if num_components == 2 else plt.axes(projection='3d')
		scatter = ax.scatter(*x_transform.T, c=y, alpha=0.5, cmap=cmap) \
			if num_components == 2 else \
			ax.scatter3D(*x_transform.T, c=y, alpha=0.5, cmap=cmap)
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
		plt.axis('scaled')
		plt.show()
