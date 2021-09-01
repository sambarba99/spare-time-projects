"""
Naive Bayes classification demo

Author: Sam Barba
Created 21/11/2021
"""

import matplotlib.pyplot as plt
from naive_bayes_classifier import NaiveBayesClassifier
import numpy as np

plt.rcParams['figure.figsize'] = (7, 5)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def extract_data(path, train_test_ratio=0.8):
	"""Split file data into train/test"""

	data = np.genfromtxt(path, dtype=str, delimiter='\n')
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
	n_classes = len(np.unique(actual))
	conf_mat = np.zeros((n_classes, n_classes)).astype(int)

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
	choice = input('Enter B to use breast tumour dataset,'
		+ '\nI for iris dataset,'
		+ '\nP for pulsar dataset,'
		+ '\nT for Titanic dataset,'
		+ '\nor W for wine dataset\n>>> ').upper()

	match choice:
		case 'B': path = 'C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\breastTumourData.txt'
		case 'I': path = 'C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\irisData.txt'
		case 'P': path = 'C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\pulsarData.txt'
		case 'T': path = 'C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\titanicData.txt'
		case _: path = 'C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\wineData.txt'

	x_train, y_train, x_test, y_test = extract_data(path)

	clf = NaiveBayesClassifier()
	clf.fit(x_train, y_train)

	# Plot confusion matrices

	train_predictions = [clf.predict(i) for i in x_train]
	test_predictions = [clf.predict(i) for i in x_test]
	train_conf_mat, train_acc = confusion_matrix(train_predictions, y_train)
	test_conf_mat, test_acc = confusion_matrix(test_predictions, y_test)

	plot_confusion_matrices(train_conf_mat, train_acc, test_conf_mat, test_acc)
