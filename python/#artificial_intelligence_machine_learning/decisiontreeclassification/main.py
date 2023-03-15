"""
Decision tree classification demo

Author: Sam Barba
Created 03/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from decision_tree import DecisionTree
from tree_plotter import plot_tree


plt.rcParams['figure.figsize'] = (8, 5)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)


def load_data(path, train_test_ratio=0.8):
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
	features = x.columns

	# Label encode y
	y = y.astype('category').cat.codes.to_frame()
	y.columns = ['classification']

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}\n')

	x, y = x.to_numpy().astype(float), y.to_numpy().squeeze().astype(int)
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio, stratify=y)

	return features, labels, x_train, y_train, x_test, y_test


def make_best_tree(x_train, y_train, x_test, y_test):
	"""Test different max_depth values, and return tree with the best one"""

	# 0 max_depth means classifying all data points as the same
	# e.g. for iris dataset, classifying them all as 0 (setosa) = 33% accuracy
	max_depth = 0
	best_tree = None
	best_test_f1 = -1

	while True:
		tree = DecisionTree(x_train, y_train, max_depth)
		train_f1 = tree.evaluate(x_train, y_train)
		test_f1 = tree.evaluate(x_test, y_test)
		print(f'max_depth {max_depth}: training F1 score = {train_f1} | test F1 score = {test_f1}')

		if test_f1 > best_test_f1:
			best_tree, best_test_f1 = tree, test_f1
		else:
			break  # No improvement, so stop

		max_depth += 1

	return best_tree


def plot_confusion_matrix(actual, predictions, labels, is_training):
	cm = confusion_matrix(actual, predictions)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
	f1 = f1_score(actual, predictions, average='binary' if len(labels) == 2 else 'weighted')

	disp.plot(cmap=plt.cm.plasma)
	plt.title(f'{"Training" if is_training else "Test"} confusion matrix\n(F1 score: {f1})')
	plt.show()


if __name__ == '__main__':
	choice = input(
		'\nEnter 1 to use banknote dataset,'
		'\n2 for breast tumour dataset,'
		'\n3 for iris dataset,'
		'\n4 for pulsar dataset,'
		'\n5 for Titanic dataset,'
		'\nor 6 for wine dataset\n>>> '
	)

	match choice:
		case '1': path = r'C:\Users\Sam\Desktop\Projects\datasets\banknoteData.csv'
		case '2': path = r'C:\Users\Sam\Desktop\Projects\datasets\breastTumourData.csv'
		case '3': path = r'C:\Users\Sam\Desktop\Projects\datasets\irisData.csv'
		case '4': path = r'C:\Users\Sam\Desktop\Projects\datasets\pulsarData.csv'
		case '5': path = r'C:\Users\Sam\Desktop\Projects\datasets\titanicData.csv'
		case _: path = r'C:\Users\Sam\Desktop\Projects\datasets\wineData.csv'

	features, labels, x_train, y_train, x_test, y_test = load_data(path)

	tree = make_best_tree(x_train, y_train, x_test, y_test)

	print(f'\nOptimal tree depth: {tree.depth}')

	plot_tree(tree, features, labels)

	train_pred = np.array([tree.predict(i) for i in x_train])
	test_pred = np.array([tree.predict(i) for i in x_test])
	plot_confusion_matrix(y_train, train_pred, labels, True)
	plot_confusion_matrix(y_test, test_pred, labels, False)
