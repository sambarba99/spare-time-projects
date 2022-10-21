"""
SVM demo

Author: Sam Barba
Created 22/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from svm import SVM

plt.rcParams['figure.figsize'] = (7, 6)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def load_data(train_test_ratio=0.8):
	df = pd.read_csv(r'C:\Users\Sam Barba\Desktop\Programs\datasets\svmData.csv')
	data = df.to_numpy()
	x, y = data[:, :-1], data[:, -1].astype(int)

	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio, stratify=y)

	return x_train, y_train, x_test, y_test

def confusion_matrix(predictions, actual):
	predictions[predictions == -1] = 0
	actual[actual == -1] = 0

	conf_mat = np.zeros((2, 2)).astype(int)

	for a, p in zip(actual, predictions):
		conf_mat[a][p] += 1

	f1 = f1_score(actual, predictions)

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

def get_hyperplane_value(x, weights, bias, offset):
	return (-weights[0] * x + bias + offset) / weights[1]

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	x_train, y_train, x_test, y_test = load_data()

	clf = SVM()
	clf.fit(x_train, y_train)

	# Plot confusion matrices

	train_predictions = clf.predict(x_train)
	test_predictions = clf.predict(x_test)
	train_conf_mat, train_f1 = confusion_matrix(train_predictions, y_train)
	test_conf_mat, test_f1 = confusion_matrix(test_predictions, y_test)

	plot_confusion_matrices(train_conf_mat, train_f1, test_conf_mat, test_f1)

	# Visualise SVM

	x_scatter = np.append(x_train, x_test, axis=0)
	y_scatter = np.append(y_train, y_test)

	for class_label in np.unique(y_scatter):
		plt.scatter(*x_scatter[y_scatter == class_label].T, alpha=0.7, label=f'Class {class_label}')

	hyperplane_start_x = np.min(x_scatter[:, 0])
	hyperplane_end_x = np.max(x_scatter[:, 0])
	hyperplane_start_y = get_hyperplane_value(hyperplane_start_x, clf.weights, clf.bias, 0)
	hyperplane_end_y = get_hyperplane_value(hyperplane_end_x, clf.weights, clf.bias, 0)
	negative_plane_start_y = get_hyperplane_value(hyperplane_start_x, clf.weights, clf.bias, -1)
	negative_plane_end_y = get_hyperplane_value(hyperplane_end_x, clf.weights, clf.bias, -1)
	positive_plane_start_y = get_hyperplane_value(hyperplane_start_x, clf.weights, clf.bias, 1)
	positive_plane_end_y = get_hyperplane_value(hyperplane_end_x, clf.weights, clf.bias, 1)

	plt.plot([hyperplane_start_x, hyperplane_end_x], [hyperplane_start_y, hyperplane_end_y], color='black', ls='--')
	plt.plot([hyperplane_start_x, hyperplane_end_x], [negative_plane_start_y, negative_plane_end_y], color='red')
	plt.plot([hyperplane_start_x, hyperplane_end_x], [positive_plane_start_y, positive_plane_end_y], color='red')

	y_min = np.min(x_scatter[:, 1])
	y_max = np.max(x_scatter[:, 1])
	plt.ylim([y_min - 0.5, y_max + 0.5])

	w = ', '.join(f'{we:.3f}' for we in clf.weights)
	m = -clf.weights[0] / clf.weights[1]
	c = -clf.bias / clf.weights[1]

	plt.axis('scaled')
	plt.title(f'Weights: {w}\nBias: {clf.bias:.3f}\nm: {m:.3f}  |  c: {c:.3f}')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
