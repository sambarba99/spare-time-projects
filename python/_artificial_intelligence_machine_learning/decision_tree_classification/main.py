"""
Decision tree classification demo

Author: Sam Barba
Created 03/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from _utils.csv_data_loader import load_csv_classification_data
from _utils.model_evaluation_plots import plot_confusion_matrix, plot_roc_curve
from decision_tree import DecisionTree
from tree_plotter import plot_tree


plt.rcParams['figure.figsize'] = (8, 5)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)


def make_best_tree(x_train, y_train, x_test, y_test):
	"""Test different max_depth values, and return tree with the best one"""

	best_tree = None
	best_test_f1 = -1
	max_depth = 0  # 0 max_depth means predicting all data points as the same value

	while True:
		tree = DecisionTree(x_train, y_train, max_depth)
		train_f1 = tree.evaluate(x_train, y_train)
		test_f1 = tree.evaluate(x_test, y_test)
		print(f'max_depth {max_depth}: training F1 score = {train_f1} | test F1 score = {test_f1}')

		if test_f1 > best_test_f1:
			best_tree, best_test_f1 = tree, test_f1
			if test_f1 == 1: break
		else:
			break  # No improvement, so stop

		max_depth += 1

	return best_tree


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
		case '1': path = 'C:/Users/Sam/Desktop/projects/datasets/banknote_authentication.csv'
		case '2': path = 'C:/Users/Sam/Desktop/projects/datasets/breast_tumour_pathology.csv'
		case '3': path = 'C:/Users/Sam/Desktop/projects/datasets/glass_classification.csv'
		case '4': path = 'C:/Users/Sam/Desktop/projects/datasets/iris_classification.csv'
		case '5': path = 'C:/Users/Sam/Desktop/projects/datasets/mushroom_edibility_classification.csv'
		case '6': path = 'C:/Users/Sam/Desktop/projects/datasets/pulsar_identification.csv'
		case '7': path = 'C:/Users/Sam/Desktop/projects/datasets/titanic_survivals.csv'
		case _: path = 'C:/Users/Sam/Desktop/projects/datasets/wine_classification.csv'

	x_train, y_train, x_test, y_test, labels, features = load_csv_classification_data(path, train_size=0.8, test_size=0.2)

	tree = make_best_tree(x_train, y_train, x_test, y_test)
	print(f'\nOptimal tree depth: {tree.depth}')

	plot_tree(tree, features, labels)

	# Confusion matrix
	test_pred = [tree.predict(i) for i in x_test]
	test_pred_classes = [p['class'] for p in test_pred]
	f1 = f1_score(y_test, test_pred_classes, average='binary' if len(labels) == 2 else 'weighted')
	plot_confusion_matrix(y_test, test_pred_classes, labels, f'Test confusion matrix\n(F1 score: {f1:.3f})')

	# ROC curve
	if len(labels) == 2:  # Binary classification
		test_pred_probs = np.array([p['class_probs'] for p in test_pred])
		plot_roc_curve(y_test, test_pred_probs[:, 1])  # Assuming 1 is the positive class
