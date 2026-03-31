"""
Decision tree regression demo

Author: Sam Barba
Created 19/10/2022
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from _utils.csv_data_loader import load_csv_regression_data
from decision_tree import DecisionTree
from tree_plotter import plot_tree


plt.rcParams['figure.figsize'] = (8, 5)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)


def make_best_tree(x_train, y_train, x_val, y_val):
	"""Tune a tree (try different max_depth values), and return tree with the best validation RMSE"""

	best_tree = None
	best_rmse = np.inf
	max_depth = 0  # 0 max_depth means predicting all data points as the same value

	while True:
		tree = DecisionTree(x_train, y_train, max_depth)
		train_rmse = tree.evaluate(x_train, y_train)
		val_rmse = tree.evaluate(x_val, y_val)
		print(f'max_depth {max_depth}: training RMSE = {train_rmse:.4f} | val RMSE = {val_rmse:.4f}')

		if val_rmse < best_rmse:
			best_tree, best_rmse = tree, val_rmse
			if val_rmse == 0:
				break
		else:
			break  # No improvement, so stop

		max_depth += 1

	return best_tree


if __name__ == '__main__':
	choice = input(
		'\nEnter B for Boston housing dataset,'
		'\nC for car value dataset,'
		'\nM for medical insurance dataset,'
		'\nP for Parkinson\'s dataset,'
		'\nor S for sine wave\n>>> '
	).upper()

	match choice:
		case 'B': path = 'C:/Users/sam/Desktop/projects/datasets/boston_housing.csv'
		case 'C': path = 'C:/Users/sam/Desktop/projects/datasets/car_valuation.csv'
		case 'M': path = 'C:/Users/sam/Desktop/projects/datasets/medical_costs.csv'
		case 'P': path = 'C:/Users/sam/Desktop/projects/datasets/parkinsons_scale.csv'
		case _: path = 'sine'

	if path == 'sine':
		x = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
		y = np.sin(x) + np.random.uniform(-0.1, 0.1, 100).reshape(-1, 1)
		x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=1)
		features = ['x']
	else:
		x_train, y_train, x_val, y_val, features = load_csv_regression_data(path, train_size=0.8, val_size=0.2)

	if path == 'sine':
		x = np.linspace(0, 2 * np.pi, 100)
		y = np.sin(x) + np.random.uniform(-0.1, 0.1, 100)

		plt.scatter(x, y, s=5, color='black', label='Data')
		for max_depth in [0, 1, 6]:
			tree = DecisionTree(x_train, y_train, max_depth)
			pred = [tree.predict([xi]) for xi in x]
			plt.plot(x, pred, label=f'Tree depth {tree.depth}')
		plt.title('Sine wave prediction with different tree depths')
		plt.legend()
		plt.show()
	else:
		tree = make_best_tree(x_train, y_train, x_val, y_val)
		print(f'\nOptimal tree depth: {tree.depth}')

	plot_tree(tree, features)
