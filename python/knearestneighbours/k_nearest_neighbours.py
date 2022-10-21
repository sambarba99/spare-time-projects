"""
KNN demo

Author: Sam Barba
Created 11/09/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from knn_classifier import KNN

plt.rcParams['figure.figsize'] = (5, 5)
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
	classes = y.unique()

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
		y = pd.get_dummies(y, prefix='class')
		# Ensure dummy column corresponds with 'classes'
		drop_idx = int(y.columns[0].endswith(classes[0]))
		y = y.drop(y.columns[drop_idx], axis=1)

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}\n')

	x, y = x.to_numpy().astype(float), np.squeeze(y.to_numpy().astype(int))
	if np.ndim(y) > 1:
		y = np.argmax(y, axis=1)

	return x, y

def confusion_matrix(predictions, actual):
	n_classes = len(np.unique(actual))
	conf_mat = np.zeros((n_classes, n_classes)).astype(int)

	for a, p in zip(actual, predictions):
		conf_mat[a][p] += 1

	f1 = f1_score(actual, predictions, average='binary' if n_classes == 2 else 'weighted')

	return conf_mat, f1

def plot_confusion_matrix(k, conf_mat, f1):
	ax = plt.subplot()
	ax.matshow(conf_mat, cmap=plt.cm.plasma)
	ax.xaxis.set_ticks_position('bottom')
	for (j, i), val in np.ndenumerate(conf_mat):
		ax.text(x=i, y=j, s=val, ha='center', va='center')
	plt.xlabel('Predictions')
	plt.ylabel('Actual')
	plt.title(f'Confusion Matrix (optimal k = {k})\nF1 score: {f1}')
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	choice = input('Enter 1 to use banknote dataset,'
		+ '\n2 for breast tumour dataset,'
		+ '\n3 for iris dataset,'
		+ '\nor 4 for wine dataset\n>>> ')

	match choice:
		case '1': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\banknoteData.csv'
		case '2': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\breastTumourData.csv'
		case '3': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\irisData.csv'
		case _: path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\wineData.csv'

	x, y = load_data(path)

	best_f1 = best_k = -1
	best_conf_mat = None

	for k in range(3, int(len(x) ** 0.5) + 1, 2):
		clf = KNN(k)
		clf.fit(x, y)

		predictions = [clf.predict(i) for i in x]
		conf_mat, f1 = confusion_matrix(predictions, y)

		print(f'F1 score with k = {k}: {f1}')

		if f1 > best_f1:
			best_f1, best_k, best_conf_mat = f1, k, conf_mat
		else:
			break  # No improvement, so stop

	plot_confusion_matrix(best_k, best_conf_mat, best_f1)

if __name__ == '__main__':
	main()
