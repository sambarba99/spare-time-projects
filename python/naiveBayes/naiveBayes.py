# Naive Bayes classification demo
# Author: Sam Barba
# Created 21/11/2021

import matplotlib.pyplot as plt
from naivebayesclassifier import NaiveBayesClassifier
import numpy as np

plt.rcParams["figure.figsize"] = (6, 6)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Split file data into train/test
def extract_data(path, train_test_ratio=0.5):
	data = np.genfromtxt(path, dtype=str, delimiter="\n")
	# Skip header and convert to floats
	data = [row.split() for row in data[1:]]
	data = np.array(data).astype(float)
	np.random.shuffle(data)

	x, y = data[:,:-1], data[:,-1].astype(int)

	split = int(len(data) * train_test_ratio)

	x_train, y_train = x[:split], y[:split]
	x_test, y_test = x[split:], y[split:]

	return x_train, y_train, x_test, y_test

def confusion_matrix(predictions, actual):
	num_classes = len(np.unique(actual))
	conf_mat = np.zeros((num_classes, num_classes)).astype(int)

	for a, p in zip(actual, predictions):
		conf_mat[a, p] += 1

	accuracy = np.trace(conf_mat) / conf_mat.sum()
	return conf_mat, accuracy

def plot_matrix(is_training, conf_mat, accuracy):
	ax = plt.subplot()
	ax.matshow(conf_mat, cmap=plt.cm.Blues, alpha=0.7)
	ax.xaxis.set_ticks_position("bottom")
	for i in range(conf_mat.shape[0]):
		for j in range(conf_mat.shape[1]):
			ax.text(x=j, y=i, s=conf_mat[i, j], ha="center", va="center")
	plt.xlabel("Predictions")
	plt.ylabel("Actual")
	title = "Training" if is_training else "Test"
	plt.title(f"{title} Confusion Matrix\nAccuracy = {accuracy}")
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
	choice = input("Enter B to use breast tumour dataset,"
		+ "\nI for iris dataset,"
		+ "\nP for pulsar dataset,"
		+ "\nT for Titanic dataset,"
		+ "\nor W for wine dataset: ").upper()

	if choice == "B":
		path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\breastTumourData.txt"
	elif choice == "I":
		path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\irisData.txt"
	elif choice == "P":
		path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\pulsarData.txt"
	elif choice == "T":
		path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\titanicData.txt"
	else:
		path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\wineData.txt"

	x_train, y_train, x_test, y_test = extract_data(path)

	clf = NaiveBayesClassifier()
	clf.fit(x_train, y_train)
	clf.train()

	# Plot confusion matrices

	train_predictions = [clf.predict(i) for i in x_train]
	test_predictions = [clf.predict(i) for i in x_test]
	train_conf_mat, train_acc = confusion_matrix(train_predictions, y_train)
	test_conf_mat, test_acc = confusion_matrix(test_predictions, y_test)

	plot_matrix(True, train_conf_mat, train_acc)
	plot_matrix(False, test_conf_mat, test_acc)
