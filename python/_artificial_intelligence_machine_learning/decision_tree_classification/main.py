"""
Decision tree classification demo

Author: Sam Barba
Created 03/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score

from _utils.csv_data_loader import load_csv_classification_data
from _utils.model_evaluation_plots import plot_confusion_matrix, plot_roc_curve
from decision_tree import DecisionTree
from tree_plotter import plot_tree


plt.rcParams['figure.figsize'] = (8, 5)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)


def make_best_tree(x_train, y_train, x_test, y_test, use_gini=True):
	"""Test different max_depth values, and return tree with the best one"""

	best_tree = None
	best_test_f1 = -1
	max_depth = 0  # 0 max_depth means predicting all data points as the same value

	while True:
		tree = DecisionTree(x_train, y_train, max_depth, use_gini)
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
		'\n8 for wine dataset,'
		'\nor 9 for blobs\n>>> '
	)

	match choice:
		case '1': path = 'C:/Users/Sam/Desktop/projects/datasets/banknote_authentication.csv'
		case '2': path = 'C:/Users/Sam/Desktop/projects/datasets/breast_tumour_pathology.csv'
		case '3': path = 'C:/Users/Sam/Desktop/projects/datasets/glass_classification.csv'
		case '4': path = 'C:/Users/Sam/Desktop/projects/datasets/iris_classification.csv'
		case '5': path = 'C:/Users/Sam/Desktop/projects/datasets/mushroom_edibility_classification.csv'
		case '6': path = 'C:/Users/Sam/Desktop/projects/datasets/pulsar_identification.csv'
		case '7': path = 'C:/Users/Sam/Desktop/projects/datasets/titanic_survivals.csv'
		case '8': path = 'C:/Users/Sam/Desktop/projects/datasets/wine_classification.csv'
		case _: path = 'blobs'

	if path == 'blobs':
		x, y = make_blobs(n_samples=500, centers=5, cluster_std=2)
		max_depth = 0
		trees, f1_scores = [], []

		while True:
			tree = DecisionTree(x, y, max_depth)
			f1 = tree.evaluate(x, y)
			trees.append(tree)
			f1_scores.append(f1)

			if len(f1_scores) > 1 and f1_scores[-1] <= f1_scores[-2]:
				trees.pop()
				f1_scores.pop()
				break  # No improvement, so stop

			max_depth += 1

		_, (ax_classification, ax_f1) = plt.subplots(ncols=2, figsize=(9, 5))

		for idx, (tree, f1) in enumerate(zip(trees, f1_scores)):
			ax_classification.clear()
			ax_f1.clear()

			ax_classification.scatter(*x.T, c=y, cmap='jet', alpha=0.7)

			# Plot mesh and mesh point classifications

			x_min, x_max = x[:, 0].min(), x[:, 0].max()
			y_min, y_max = x[:, 1].min(), x[:, 1].max()
			xx, yy = np.meshgrid(
				np.linspace(x_min - 0.1, x_max + 0.1, 500),
				np.linspace(y_min - 0.1, y_max + 0.1, 500)
			)
			mesh_coords = np.column_stack((xx.flatten(), yy.flatten()))
			mesh_y = np.array([tree.predict(xi)['class'] for xi in mesh_coords])
			mesh_y = mesh_y.reshape(xx.shape)
			ax_classification.imshow(
				mesh_y, interpolation='nearest', cmap='jet', alpha=0.2, aspect='auto', origin='lower',
				extent=(xx.min(), xx.max(), yy.min(), yy.max())
			)

			ax_f1.plot(f1_scores[:idx + 1])
			ax_f1.set_xlabel('Max tree depth')
			ax_f1.set_ylabel('F1')
			ax_f1.set_title('F1 score')

			if idx < len(trees) - 1:
				ax_classification.set_title(f'Max depth = {tree.depth}, F1 score = {f1:.4f}')
				plt.draw()
				plt.pause(1)
			else:
				ax_classification.set_title(f'Max depth = {tree.depth}, F1 score = {f1:.4f}\n(converged)')
				plt.show()
	else:
		x_train, y_train, x_test, y_test, labels, features = load_csv_classification_data(path, train_size=0.8, test_size=0.2)

		tree = make_best_tree(x_train, y_train, x_test, y_test)
		print(f'\nOptimal tree depth: {tree.depth}')

		plot_tree(tree, features, labels)

		# Confusion matrix
		test_pred = [tree.predict(xi) for xi in x_test]
		test_pred_classes = [p['class'] for p in test_pred]
		f1 = f1_score(y_test, test_pred_classes, average='binary' if len(labels) == 2 else 'weighted')
		plot_confusion_matrix(y_test, test_pred_classes, labels, f'Test confusion matrix\n(F1 score: {f1:.3f})')

		# ROC curve
		if len(labels) == 2:  # Binary classification
			test_pred_probs = np.array([p['class_probs'] for p in test_pred])
			plot_roc_curve(y_test, test_pred_probs[:, 1])  # Assuming 1 is the positive class
