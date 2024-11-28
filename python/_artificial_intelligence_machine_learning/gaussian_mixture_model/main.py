"""
Gaussian Mixture Model demo

Author: Sam Barba
Created 15/12/2023
"""

import numpy as np
import pandas as pd

from _utils.csv_data_loader import load_csv_classification_data
from gaussian_mixture_model import GaussianMixtureModel


pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)


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

	x_train, y_train, x_test, y_test, *_ = load_csv_classification_data(path, train_size=0.8, test_size=0.2)

	gmm = GaussianMixtureModel(num_components=len(np.unique(y_train)))
	gmm.fit(x_train)
	predictions = gmm.predict(x_test)

	print('\nTest set stats:')
	for label, count in zip(*np.unique(y_test, return_counts=True)):
		print(f'\tLabel: {label} | count: {count}')

	print('Prediction stats:')
	for label, count in zip(*np.unique(predictions, return_counts=True)):
		print(f'\tLabel: {label} | count: {count}')
