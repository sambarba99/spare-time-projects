"""
Perceptron demo

Author: Sam Barba
Created 23/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
from perceptronclassifier import PerceptronClf

plt.rcParams['figure.figsize'] = (7, 6)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def extract_data(train_test_ratio=0.8):
	"""Split file data into train/test"""

	data = np.genfromtxt('C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\svmData.txt',
		dtype=str, delimiter='\n')
	# Skip header and convert to floats
	data = [row.split() for row in data[1:]]
	data = np.array(data).astype(float)
	np.random.shuffle(data)

	x, y = data[:, :-1], data[:, -1].astype(int)
	# File data is for SVM testing, so convert class -1 to 0
	y[y == -1] = 0

	split = int(len(data) * train_test_ratio)

	x_train, y_train = x[:split], y[:split]
	x_test, y_test = x[split:], y[split:]

	return x_train, y_train, x_test, y_test

def confusion_matrix(predictions, actual):
	num_classes = len(np.unique(actual))
	conf_mat = np.zeros((num_classes, num_classes)).astype(int)

	for a, p in zip(actual, predictions):
		conf_mat[a][p] += 1

	accuracy = np.trace(conf_mat) / conf_mat.sum()
	return conf_mat, accuracy

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
	axes[0].set_title(f'Training Confusion Matrix\nAccuracy = {train_acc:.3f}')
	axes[1].set_title(f'Test Confusion Matrix\nAccuracy = {test_acc:.3f}')
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	x_train, y_train, x_test, y_test = extract_data()

	clf = PerceptronClf()
	clf.fit(x_train, y_train)
	clf.train()

	# Plot confusion matrices

	train_predictions = clf.predict(x_train)
	test_predictions = clf.predict(x_test)
	train_conf_mat, train_acc = confusion_matrix(train_predictions, y_train)
	test_conf_mat, test_acc = confusion_matrix(test_predictions, y_test)

	plot_confusion_matrices(train_conf_mat, train_acc, test_conf_mat, test_acc)

	# Visualise perceptron

	x_scatter = np.append(x_train, x_test, axis=0)
	y_scatter = np.append(y_train, y_test)

	for class_label in np.unique(y_scatter):
		plt.scatter(*x_scatter[y_scatter == class_label].T, alpha=0.7, label=f'Class {class_label}')

	decision_bound_x1 = np.min(x_scatter[:, 0])
	decision_bound_x2 = np.max(x_scatter[:, 0])
	decision_bound_y1 = (-clf.weights[0] * decision_bound_x1 - clf.bias) / clf.weights[1]
	decision_bound_y2 = (-clf.weights[0] * decision_bound_x2 - clf.bias) / clf.weights[1]

	plt.plot([decision_bound_x1, decision_bound_x2], [decision_bound_y1, decision_bound_y2],
		color='black', ls='--')

	y_min = np.min(x_scatter[:, 1])
	y_max = np.max(x_scatter[:, 1])
	plt.ylim([y_min - 0.5, y_max + 0.5])

	w = ', '.join(f'{we:.3f}' for we in clf.weights)
	m = -clf.weights[0] / clf.weights[1]
	c = -clf.bias / clf.weights[1]

	plt.axis('scaled')
	plt.title(f'Weights: {w}\nBias: {clf.bias:.3f}\nm: {m:.3f} | c: {c:.3f}')
	plt.legend()
	plt.show()
