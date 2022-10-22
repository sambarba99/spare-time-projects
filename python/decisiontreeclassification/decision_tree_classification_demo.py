"""
Decision tree classification demo

Author: Sam Barba
Created 03/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from decision_tree import DecisionTree
from tree_plotter import plot_tree

plt.rcParams['figure.figsize'] = (7, 4)
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
	classes = sorted(y.unique())

	for col in x_to_encode:
		if len(x[col].unique()) > 2:
			one_hot = pd.get_dummies(x[col], prefix=col)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)
		else:  # Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True)
	features = x.columns

	if len(classes) > 2:
		one_hot = pd.get_dummies(y, prefix='class')
		y = pd.concat([y, one_hot], axis=1)
		y = y.drop(y.columns[0], axis=1)
	else:  # Binary class
		y = pd.get_dummies(y, prefix='class', drop_first=True)

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}\n')

	x, y = x.to_numpy().astype(float), np.squeeze(y.to_numpy().astype(int))
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio, stratify=y)

	return features, classes, x_train, y_train, x_test, y_test

def make_best_tree(x_train, y_train, x_test, y_test):
	"""Test different max_depth values, and return tree with the best one"""

	# 0 max_depth means classifying all data points as the same
	# e.g. for iris dataset, classifying them all as 0 (setosa) = 33% accuracy
	max_depth = 0
	best_tree = None
	best_mean_f1 = -1

	while True:
		tree = DecisionTree(x_train, y_train, max_depth)
		train_f1 = tree.evaluate(x_train, y_train)
		test_f1 = tree.evaluate(x_test, y_test)
		mean_f1 = (train_f1 + test_f1) / 2
		print(f'max_depth {max_depth}: training F1 score = {train_f1} | test F1 score = {test_f1} | mean = {mean_f1}')

		if mean_f1 > best_mean_f1:
			best_tree, best_mean_f1 = tree, mean_f1
		else:
			break  # No improvement, so stop

		max_depth += 1

	return best_tree

def confusion_matrix(predictions, actual):
	if np.ndim(actual) > 1:
		actual = np.argmax(actual, axis=1)
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

	features, classes, x_train, y_train, x_test, y_test = load_data(path)

	tree = make_best_tree(x_train, y_train, x_test, y_test)

	print(f'\nOptimal tree depth: {tree.get_depth()}')

	plot_tree(tree, features, classes)

	# Plot confusion matrices

	train_predictions = [tree.predict(i) for i in x_train]
	test_predictions = [tree.predict(i) for i in x_test]
	train_conf_mat, train_f1 = confusion_matrix(train_predictions, y_train)
	test_conf_mat, test_f1 = confusion_matrix(test_predictions, y_test)

	plot_confusion_matrices(train_conf_mat, train_f1, test_conf_mat, test_f1)
