"""
SVM demo

Author: Sam Barba
Created 22/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
from svm import SVM

plt.rcParams['figure.figsize'] = (7, 6)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def extract_data(train_test_ratio=0.8):
	"""Split file data into train/test"""

	data = np.genfromtxt(r'C:\Users\Sam Barba\Desktop\Programs\datasets\svmData.txt', dtype=str, delimiter='\n')
	# Skip header and convert to floats
	data = [row.split() for row in data[1:]]
	data = np.array(data).astype(float)
	np.random.shuffle(data)

	x, y = data[:, :-1], data[:, -1].astype(int)

	split = int(len(data) * train_test_ratio)

	x_train, y_train = x[:split], y[:split]
	x_test, y_test = x[split:], y[split:]

	return x_train, y_train, x_test, y_test

def confusion_matrix(predictions, actual):
	predictions[predictions == -1] = 0
	actual[actual == -1] = 0

	n_classes = len(np.unique(actual))
	conf_mat = np.zeros((n_classes, n_classes)).astype(int)

	for a, p in zip(actual, predictions):
		conf_mat[a][p] += 1

	# True positive, false positive, false negative
	tp = conf_mat[1][1]
	fp = conf_mat[0][1]
	fn = conf_mat[1][0]
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1_score = 2 * (precision * recall) / (precision + recall)

	return conf_mat, f1_score

def plot_confusion_matrices(train_conf_mat, train_acc, test_conf_mat, test_acc):
	# axes[0] = training confusion matrix
	# axes[1] = test confusion matrix
	_, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
	axes[0].matshow(train_conf_mat, cmap=plt.cm.Blues, alpha=0.7)
	axes[1].matshow(test_conf_mat, cmap=plt.cm.Blues, alpha=0.7)
	axes[0].xaxis.set_ticks_position('bottom')
	axes[1].xaxis.set_ticks_position('bottom')
	for (j, i), val in np.ndenumerate(train_conf_mat):
		axes[0].text(x=i, y=j, s=val, ha='center', va='center')
		axes[1].text(x=i, y=j, s=test_conf_mat[j][i], ha='center', va='center')
	axes[0].set_xlabel('Predictions')
	axes[0].set_ylabel('Actual')
	axes[0].set_title(f'Training Confusion Matrix\nF1 score = {train_acc:.3f}')
	axes[1].set_title(f'Test Confusion Matrix\nF1 score = {test_acc:.3f}')
	plt.show()

def get_hyperplane_value(x, weights, bias, offset):
	return (-weights[0] * x + bias + offset) / weights[1]

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	x_train, y_train, x_test, y_test = extract_data()

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
