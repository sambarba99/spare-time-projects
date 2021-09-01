"""
KNN demo

Author: Sam Barba
Created 11/09/2021
"""

from knn_classifier import KNN
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (6, 6)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def confusion_matrix(predictions, actual):
	n_classes = len(np.unique(actual))
	conf_mat = np.zeros((n_classes, n_classes)).astype(int)

	for a, p in zip(actual, predictions):
		conf_mat[a][p] += 1

	accuracy = np.trace(conf_mat) / conf_mat.sum()
	return conf_mat, accuracy

def plot_matrix(k, conf_mat, accuracy):
	ax = plt.subplot()
	ax.matshow(conf_mat, cmap=plt.cm.Blues, alpha=0.7)
	ax.xaxis.set_ticks_position('bottom')
	for (j, i), val in np.ndenumerate(conf_mat):
		ax.text(x=i, y=j, s=val, ha='center', va='center')
	plt.xlabel('Predictions')
	plt.ylabel('Actual')
	plt.title(f'Confusion Matrix (optimal k = {k})\nAccuracy = {accuracy:.3f}')
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	choice = input('Enter I to use iris dataset or W for wine dataset\n>>> ').upper()
	print()
	path = 'C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\irisData.txt' if choice == 'I' \
		else 'C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\wineData.txt'

	data = np.genfromtxt(path, dtype=str, delimiter='\n')
	# Skip header and convert to floats
	data = [row.split() for row in data[1:]]
	data = np.array(data).astype(float)
	np.random.shuffle(data)

	x, y = data[:, :-1], data[:, -1].astype(int)

	best_acc = best_k = -1
	best_conf_mat = None

	for k in range(3, int(len(data) ** 0.5) + 1, 2):
		clf = KNN(k)
		clf.fit(x, y)

		predictions = [clf.predict(i) for i in x]
		conf_mat, acc = confusion_matrix(predictions, y)

		if acc > best_acc:
			best_acc, best_k, best_conf_mat = acc, k, conf_mat

		print(f'Accuracy with k = {k}: {acc}')

	plot_matrix(best_k, best_conf_mat, best_acc)
