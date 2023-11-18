"""
Gaussian Mixture Model demo

Author: Sam Barba
Created 15/12/2023
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from gaussian_mixture_model import GaussianMixtureModel


np.random.seed(1)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)


def load_data(path, train_test_ratio=0.8):
	df = pd.read_csv(path)
	print(f'\nRaw data:\n\n{df}')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]
	x_to_encode = x.select_dtypes(exclude=np.number).columns

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

	print(f'\nPreprocessed data:\n\n{pd.concat([x, y], axis=1)}\n')

	x, y = x.to_numpy(), y.to_numpy().squeeze()
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio, stratify=y, random_state=1)

	return x_train, y_train, x_test, y_test


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

	x_train, y_train, x_test, y_test = load_data(path)

	gmm = GaussianMixtureModel(n_components=len(np.unique(y_train)))
	gmm.fit(x_train)
	predictions = gmm.predict(x_test)

	print('\nTest set stats:')
	for label, count in zip(*np.unique(y_test, return_counts=True)):
		print(f'\tLabel: {label} | count: {count}')

	print('Prediction stats:')
	for label, count in zip(*np.unique(predictions, return_counts=True)):
		print(f'\tLabel: {label} | count: {count}')
