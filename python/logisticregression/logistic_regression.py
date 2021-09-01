"""
Logistic regression demo

Author: Sam Barba
Created 10/11/2021
"""

from logistic_regressor import LogisticRegressor
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (8, 6)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def extract_data(path, train_test_ratio=0.8):
	"""Split file data into train/test"""

	data = np.genfromtxt(path, dtype=str, delimiter='\n')
	feature_names = data[0].strip().split(',')
	# Skip header and convert to floats
	data = [row.split() for row in data[1:]]
	data = np.array(data).astype(float)
	np.random.shuffle(data)

	x, y = data[:, :-1], data[:, -1].astype(int)

	# Standardise data (column-wise) (no need for y)
	split = int(len(data) * train_test_ratio)
	training_mean = np.mean(x[:split], axis=0)
	training_std = np.std(x[:split], axis=0)
	x = (x - training_mean) / training_std

	x_train, y_train = x[:split], y[:split]
	x_test, y_test = x[split:], y[split:]

	return feature_names, x_train, y_train, x_test, y_test, data

def confusion_matrix(predictions, actual):
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

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	choice = input('Enter B to use breast tumour dataset,'
		+ '\nP for pulsar dataset,'
		+ '\nor T for Titanic dataset\n>>> ').upper()

	match choice:
		case 'B':
			path = 'C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\breastTumourData.txt'
			classes = ['malignant', 'benign']
		case 'P':
			path = 'C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\pulsarData.txt'
			classes = ['not pulsar', 'pulsar']
		case _:
			path = 'C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\titanicData.txt'
			classes = ['did not survive', 'survived']

	feature_names, x_train, y_train, x_test, y_test, data = extract_data(path)

	regressor = LogisticRegressor()
	regressor.fit(x_train, y_train)

	# Plot confusion matrices

	train_conf_mat, train_f1 = confusion_matrix(regressor.predict(x_train), y_train)
	test_conf_mat, test_f1 = confusion_matrix(regressor.predict(x_test), y_test)

	plot_confusion_matrices(train_conf_mat, train_f1, test_conf_mat, test_f1)

	# Plot regression line using 2 columns with the strongest correlation with y (class)

	corr_coeffs = np.corrcoef(data.T)
	# Make bottom-right coefficient 0, as this doesn't count (correlation of last column with itself)
	corr_coeffs[-1, -1] = 0

	# Indices of columns in descending order of correlation with y
	indices = np.argsort(np.abs(corr_coeffs[:, -1]))[::-1]
	# Keep 2 strongest
	indices = indices[:2]
	max_corr = corr_coeffs[indices, -1]

	print(f"\nHighest (abs) correlation with y (class): {max_corr[0]}  (feature '{feature_names[indices[0]]}')")
	print(f"2nd highest (abs) correlation with y (class): {max_corr[1]}  (feature '{feature_names[indices[1]]}')")

	w1, w2 = regressor.weights[indices]
	m = -w1 / w2
	c = -regressor.bias / w2
	x_scatter = np.append(x_train[:, indices], x_test[:, indices], axis=0)
	y_scatter = np.append(y_train, y_test)
	x_plot = np.array([np.min(x_scatter, axis=0)[0], np.max(x_scatter, axis=0)[0]])
	y_plot = m * x_plot + c

	for idx, class_label in enumerate(np.unique(y_scatter)):
		plt.scatter(*x_scatter[y_scatter == class_label].T, alpha=0.7, label=classes[idx])
	plt.plot(x_plot, y_plot, color='black', ls='--')
	plt.ylim(np.min(x_scatter) * 1.1, np.max(x_scatter) * 1.1)
	plt.xlabel(feature_names[indices[0]] + ' (standardised)')
	plt.ylabel(feature_names[indices[1]] + ' (standardised)')
	plt.title(f'Gradient descent solution\nm = {m:.3f}  |  c = {c:.3f}')
	plt.legend()
	plt.show()

	# Plot cost graph

	plt.plot(regressor.cost_history, color='red')
	plt.xlabel('Training iteration')
	plt.ylabel('Cost')
	plt.title('Cost during training')
	plt.show()

if __name__ == '__main__':
	main()
