"""
Naive Bayes classification demo

Author: Sam Barba
Created 21/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from naive_bayes_classifier import NaiveBayesClassifier

plt.rcParams['figure.figsize'] = (7, 5)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def load_data(path, train_test_ratio=0.8):
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

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}')

	x, y = x.to_numpy().astype(float), np.squeeze(y.to_numpy().astype(int))
	if np.ndim(y) > 1:
		y = np.argmax(y, axis=1)

	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio, stratify=y)

	return x_train, y_train, x_test, y_test

def confusion_matrix(predictions, actual):
	n_classes = len(np.unique(actual))
	conf_mat = np.zeros((n_classes, n_classes)).astype(int)

	for a, p in zip(actual, predictions):
		conf_mat[a][p] += 1

	f1 = f1_score(actual, predictions, average='binary' if n_classes == 2 else 'weighted')

	return conf_mat, f1

def plot_confusion_matrices(train_conf_mat, train_f1, test_conf_mat, test_f1):
	_, (train, test) = plt.subplots(ncols=2, sharex=True, sharey=True)
	train.matshow(train_conf_mat, cmap=plt.cm.plasma)
	test.matshow(test_conf_mat, cmap=plt.cm.plasma)
	train.xaxis.set_ticks_position('bottom')
	test.xaxis.set_ticks_position('bottom')
	for (j, i), val in np.ndenumerate(train_conf_mat):
		train.text(x=i, y=j, s=val, ha='center', va='center')
		test.text(x=i, y=j, s=test_conf_mat[j][i], ha='center', va='center')
	train.set_xlabel('Predictions')
	train.set_ylabel('Actual')
	train.set_title(f'Training Confusion Matrix\nF1 score: {train_f1}')
	test.set_title(f'Test Confusion Matrix\nF1 score: {test_f1}')
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	choice = input('Enter 1 to use banknote dataset,'
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

	x_train, y_train, x_test, y_test = load_data(path)

	clf = NaiveBayesClassifier()
	clf.fit(x_train, y_train)

	# Plot confusion matrices

	train_predictions = [clf.predict(i) for i in x_train]
	test_predictions = [clf.predict(i) for i in x_test]
	train_conf_mat, train_f1 = confusion_matrix(train_predictions, y_train)
	test_conf_mat, test_f1 = confusion_matrix(test_predictions, y_test)

	plot_confusion_matrices(train_conf_mat, train_f1, test_conf_mat, test_f1)
