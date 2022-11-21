"""
KNN demo

Author: Sam Barba
Created 11/09/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score

from knn_classifier import KNN

plt.rcParams['figure.figsize'] = (8, 5)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)

def load_data(path):
	df = pd.read_csv(path)
	print(f'\nRaw data:\n{df}')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]
	x_to_encode = x.select_dtypes(exclude=np.number).columns
	labels = sorted(y.unique())

	for col in x_to_encode:
		if len(x[col].unique()) > 2:
			one_hot = pd.get_dummies(x[col], prefix=col)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)
		else:  # Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True)

	# Label encode y
	y = y.astype('category').cat.codes.to_frame()
	y.columns = ['classification']

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}\n')

	x, y = x.to_numpy().astype(float), y.to_numpy().squeeze().astype(int)
	x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))  # Normalise

	return x, y, labels

def plot_confusion_matrix(k, actual, predictions, labels):
	cm = confusion_matrix(actual, predictions)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
	f1 = f1_score(actual, predictions, average='binary' if len(labels) == 2 else 'weighted')

	disp.plot(cmap=plt.cm.plasma)
	plt.title(f'Confusion matrix for k = {k}\n(F1 score: {f1})')
	plt.show()

	return f1

if __name__ == '__main__':
	choice = input('\nEnter 1 to use banknote dataset,'
		+ '\n2 for breast tumour dataset,'
		+ '\n3 for iris dataset,'
		+ '\n4 for Titanic dataset,'
		+ '\nor 5 for wine dataset\n>>> ')

	match choice:
		case '1': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\banknoteData.csv'
		case '2': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\breastTumourData.csv'
		case '3': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\irisData.csv'
		case '4': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\titanicData.csv'
		case _: path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\wineData.csv'

	x, y, labels = load_data(path)
	best_f1 = best_k = -1

	for k in range(3, int(len(x) ** 0.5) + 1, 2):
		clf = KNN(k)
		clf.fit(x, y)

		predictions = np.array([clf.predict(i) for i in x])
		f1 = plot_confusion_matrix(k, y, predictions, labels)

		if f1 > best_f1:
			best_f1, best_k = f1, k
			if best_f1 == 1: break
		else:
			break  # No improvement, so stop

	print('Best k found:', best_k)
