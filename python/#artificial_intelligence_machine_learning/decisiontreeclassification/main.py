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
from sklearn.metrics import roc_curve
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
		n_unique = x[col].nunique()
		if n_unique == 1:
			# No information from this feature
			x = x.drop(col, axis=1)
		elif n_unique > 2:
			# Multivariate feature
			one_hot = pd.get_dummies(x[col], prefix=col)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)
		else:
			# Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True)
	features = x.columns

	# Label encode y
	y = y.astype('category').cat.codes.to_frame()
	y.columns = ['classification']

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}\n')

	x, y = x.to_numpy(), y.to_numpy().squeeze()
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio, stratify=y, random_state=1)

	return x_train, y_train, x_test, y_test, features, labels


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
		case '1': path = r'C:\Users\Sam\Desktop\Projects\datasets\banknote_authentication.csv'
		case '2': path = r'C:\Users\Sam\Desktop\Projects\datasets\breast_tumour_pathology.csv'
		case '3': path = r'C:\Users\Sam\Desktop\Projects\datasets\glass_classification.csv'
		case '4': path = r'C:\Users\Sam\Desktop\Projects\datasets\iris_classification.csv'
		case '5': path = r'C:\Users\Sam\Desktop\Projects\datasets\mushroom_edibility_classification.csv'
		case '6': path = r'C:\Users\Sam\Desktop\Projects\datasets\pulsar_identification.csv'
		case '7': path = r'C:\Users\Sam\Desktop\Projects\datasets\titanic_survivals.csv'
		case _: path = r'C:\Users\Sam\Desktop\Projects\datasets\wine_classification.csv'

	x_train, y_train, x_test, y_test, features, labels = load_data(path)

	tree = make_best_tree(x_train, y_train, x_test, y_test)
	print(f'\nOptimal tree depth: {tree.depth}')

	plot_tree(tree, features, labels)

	# Confusion matrix

	test_pred = [tree.predict(i) for i in x_test]
	test_pred_classes = [p['class'] for p in test_pred]
	cm = confusion_matrix(y_test, test_pred_classes)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
	f1 = f1_score(y_test, test_pred_classes, average='binary' if len(labels) == 2 else 'weighted')

	disp.plot(cmap=plt.cm.plasma)
	plt.title(f'Test confusion matrix\n(F1 score: {f1})')
	plt.show()

	# ROC curve

	if len(labels) == 2:  # Binary classification
		test_pred_probs = np.array([p['class_probs'] for p in test_pred])
		fpr, tpr, _ = roc_curve(y_test, test_pred_probs[:, 1])  # Assuming 1 is the positive class
		plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
		plt.plot(fpr, tpr)
		plt.axis('scaled')
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC curve')
		plt.show()
