"""
KNN demo

Author: Sam Barba
Created 11/09/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

from _utils.csv_data_loader import load_csv_classification_data
from _utils.model_evaluation_plots import plot_confusion_matrix
from knn_classifier import KNN


plt.rcParams['figure.figsize'] = (6, 4)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)


if __name__ == '__main__':
	choice = input(
		'\nEnter 1 to use banknote dataset,'
		'\n2 for breast tumour dataset,'
		'\n3 for glass dataset,'
		'\n4 for iris dataset,'
		'\n5 for Titanic dataset,'
		'\nor 6 for wine dataset\n>>> '
	)

	match choice:
		case '1': path = 'C:/Users/Sam/Desktop/projects/datasets/banknote_authenticity.csv'
		case '2': path = 'C:/Users/Sam/Desktop/projects/datasets/breast_tumour_pathology.csv'
		case '3': path = 'C:/Users/Sam/Desktop/projects/datasets/glass_classification.csv'
		case '4': path = 'C:/Users/Sam/Desktop/projects/datasets/iris_classification.csv'
		case '5': path = 'C:/Users/Sam/Desktop/projects/datasets/titanic_survivals.csv'
		case _: path = 'C:/Users/Sam/Desktop/projects/datasets/wine_classification.csv'

	x, y, labels, _ = load_csv_classification_data(path, x_transform=MinMaxScaler())
	best_f1 = best_k = -1

	for k in range(3, int(len(x) ** 0.5) + 1, 2):
		clf = KNN(k)
		clf.fit(x, y)

		predictions = np.array([clf.predict(xi) for xi in x])
		f1 = f1_score(y, predictions, average='binary' if len(labels) == 2 else 'weighted')

		print(f'F1 score for k = {k}: {f1}')

		if f1 > best_f1:
			best_f1, best_k = f1, k
			if best_f1 == 1:
				break
		else:
			break  # No improvement, so stop

	print('\nBest k found:', best_k)

	clf = KNN(best_k)
	clf.fit(x, y)
	predictions = np.array([clf.predict(xi) for xi in x])

	plot_confusion_matrix(
		y,
		predictions,
		labels,
		f'Confusion matrix for k = {best_k}\n(F1 score: {best_f1:.3f})',
		x_ticks_rotation=45,
		horiz_alignment='right'
	)
