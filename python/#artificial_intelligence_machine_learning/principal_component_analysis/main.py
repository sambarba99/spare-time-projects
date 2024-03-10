"""
Principal Component Analysis demo

Author: Sam Barba
Created 10/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


plt.rcParams['figure.figsize'] = (7, 7)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)


def load_data(path):
	df = pd.read_csv(path)
	print(f'\nRaw data:\n\n{df}')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]
	x_to_encode = x.select_dtypes(exclude=np.number).columns
	labels = sorted(y.unique())

	for col in x_to_encode:
		n_unique = x[col].nunique()
		if n_unique == 1:
			# No information from this feature
			x = x.drop(col, axis=1)
		elif n_unique == 2:
			# Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True).astype(int)
		else:
			# Multivariate feature
			one_hot = pd.get_dummies(x[col], prefix=col).astype(int)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)

	label_encoder = LabelEncoder()
	y = pd.DataFrame(label_encoder.fit_transform(y), columns=['classification'])

	print(f'\nPreprocessed data:\n\n{pd.concat([x, y], axis=1)}')

	x, y = x.to_numpy(), y.to_numpy().squeeze()
	scaler = MinMaxScaler()
	x = scaler.fit_transform(x)  # Normalise (PCA is sensitive to feature scales)

	return x, y, labels


def transform(x, n_components):
	x -= x.mean(axis=0)

	covariance = np.cov(x.T)
	eigenvalues, eigenvectors = np.linalg.eig(covariance)
	if isinstance(eigenvalues[0], np.complex128):
		eigenvalues = eigenvalues.real

	indices = eigenvalues.argsort()[::-1]
	eigenvalues = eigenvalues[indices]
	eigenvectors = eigenvectors.T[indices]

	components = eigenvectors[:n_components]
	x_transform = x.dot(components.T)

	variability = covariance.trace()
	total_explained_variance = (eigenvalues / variability)[:n_components].sum()

	return x_transform, total_explained_variance


if __name__ == '__main__':
	choice = input(
		'\nEnter 1 to use banknote dataset,'
		'\n2 for breast tumour dataset,'
		'\n3 for glass dataset,'
		'\n4 for iris dataset,'
		'\n5 for mushroom dataset,'
		'\n6 for pulsar dataset,'
		'\n7 for Titanic dataset,'
		'\nor 8 for wine dataset\n>>> '
	)

	match choice:
		case '1': path = 'C:/Users/Sam/Desktop/Projects/datasets/banknote_authentication.csv'
		case '2': path = 'C:/Users/Sam/Desktop/Projects/datasets/breast_tumour_pathology.csv'
		case '3': path = 'C:/Users/Sam/Desktop/Projects/datasets/glass_classification.csv'
		case '4': path = 'C:/Users/Sam/Desktop/Projects/datasets/iris_classification.csv'
		case '5': path = 'C:/Users/Sam/Desktop/Projects/datasets/mushroom_edibility_classification.csv'
		case '6': path = 'C:/Users/Sam/Desktop/Projects/datasets/pulsar_identification.csv'
		case '7': path = 'C:/Users/Sam/Desktop/Projects/datasets/titanic_survivals.csv'
		case _: path = 'C:/Users/Sam/Desktop/Projects/datasets/wine_classification.csv'

	n_components = None
	while n_components not in ('2', '3'):
		n_components = input('\nEnter no. components (2 or 3)\n>>> ')
	n_components = int(n_components)

	x, y, labels = load_data(path)
	x_transform, explained_variance_ratio = transform(x, n_components)

	ax = plt.axes() if n_components == 2 else plt.axes(projection='3d')
	scatter = ax.scatter(*x_transform.T, c=y, alpha=0.5, cmap='brg') \
		if n_components == 2 else \
		ax.scatter3D(*x_transform.T, c=y, alpha=0.5, cmap='brg')
	ax.set_xlabel('Principal component 1')
	ax.set_ylabel('Principal component 2')
	if n_components == 3:
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
