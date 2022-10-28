"""
PCA demo

Author: Sam Barba
Created 10/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['figure.figsize'] = (7, 7)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def load_data(path):
	df = pd.read_csv(path)
	print(f'\nRaw data:\n{df}')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]
	x_to_encode = x.select_dtypes(exclude=np.number).columns
	classes = sorted(y.unique())

	for col in x_to_encode:
		if len(x[col].unique()) > 2:
			one_hot = pd.get_dummies(x[col], prefix=col)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)
		else:  # Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True)

	if len(classes) > 2:
		one_hot = pd.get_dummies(y, prefix='class')
		y = pd.concat([y, one_hot], axis=1)
		y = y.drop(y.columns[0], axis=1)
	else:  # Binary class
		y = pd.get_dummies(y, prefix='class', drop_first=True)

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}')

	x, y = x.to_numpy().astype(float), y.squeeze().to_numpy().astype(int)
	x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))  # Normalise

	return classes, x, y

def transform(x, n_components=2):
	x -= x.mean(axis=0)

	covariance = np.cov(x.T)
	variability = covariance.trace()

	eigenvalues, eigenvectors = np.linalg.eig(covariance)
	eigenvectors = eigenvectors.T
	indices = eigenvalues.argsort()[::-1]
	eigenvalues = eigenvalues[indices]
	eigenvectors = eigenvectors[indices]

	components = eigenvectors[:n_components]

	pca_variability = (eigenvalues / variability)[:n_components].sum()

	return x.dot(components.T), pca_variability

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	choice = input('\nEnter 1 to use banknote dataset,'
		+ '\n2 for breast tumour dataset,'
		+ '\n3 for iris dataset,'
		+ '\n4 for pulsar dataset,'
		+ '\n5 for Titanic dataset,'
		+ '\nor 6 for wine dataset\n>>> ')

	match choice:
		case '1': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\banknoteData.csv'
		case '2': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\breastTumourData.csv'
		case '3': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\irisData.csv'
		case '4': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\pulsarData.csv'
		case '5': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\titanicData.csv'
		case _: path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\wineData.csv'

	classes, x, y = load_data(path)
	x_transform, new_variability = transform(x)
	if np.ndim(y) > 1:
		y = y.argmax(axis=1)  # Decode from one-hot (so can plot with c=y)

	scatter = plt.scatter(*x_transform.T, c=y, alpha=0.7, cmap=plt.cm.brg)
	plt.xlabel('Principal component 1')
	plt.ylabel('Principal component 2')
	plt.title(fr'Shape of $x$: {x.shape}'
		f'\nShape of PCA transform: {x_transform.shape}'
		f'\nCaptured variability: {new_variability}')
	handles, _ = scatter.legend_elements()
	plt.legend(handles, classes)
	plt.show()
