"""
Logistic regression demo

Author: Sam Barba
Created 10/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from logistic_regressor import LogisticRegressor

plt.rcParams['figure.figsize'] = (8, 6)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def load_data(path, train_test_ratio=0.8):
	df = pd.read_csv(path)
	print(f'\nRaw data:\n{df}')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]
	x_to_encode = x.select_dtypes(exclude=np.number).columns
	classes = y.unique()

	for col in x_to_encode:
		if len(x[col].unique()) > 2:
			one_hot = pd.get_dummies(x[col], prefix=col)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)
		else:  # Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True)
	features = x.columns

	if len(classes) > 2:
		one_hot = pd.get_dummies(y, prefix='class')
		y = pd.concat([y, one_hot], axis=1)
		y = y.drop(y.columns[0], axis=1)
	else:  # Binary class
		y = pd.get_dummies(y, prefix='class')
		# Ensure dummy column corresponds with 'classes'
		drop_idx = int(y.columns[0].endswith(classes[0]))
		y = y.drop(y.columns[drop_idx], axis=1)
		if y.iloc[0][0] == 1:
			# classes[0] = no/false/0
			classes = classes[::-1]

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}')

	data = pd.concat([x, y], axis=1).to_numpy()
	x, y = x.to_numpy().astype(float), np.squeeze(y.to_numpy().astype(int))
	if np.ndim(y) > 1:
		y = np.argmax(y, axis=1)

	# Standardise x (numeric features only)
	numeric_feature_indices = [idx for idx, f in enumerate(features) if f not in x_to_encode]
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio, stratify=y)
	training_mean = np.mean(x_train[:, numeric_feature_indices], axis=0)
	training_std = np.std(x_train[:, numeric_feature_indices], axis=0)
	x_train[:, numeric_feature_indices] = (x_train[:, numeric_feature_indices] - training_mean) / training_std
	x_test[:, numeric_feature_indices] = (x_test[:, numeric_feature_indices] - training_mean) / training_std

	return features, classes, x_train, y_train, x_test, y_test, data

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

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	choice = input('Enter B to use breast tumour dataset,'
		+ '\nP for pulsar dataset,'
		+ '\nor T for Titanic dataset\n>>> ').upper()

	match choice:
		case 'B': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\breastTumourData.csv'
		case 'P': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\pulsarData.csv'
		case _: path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\titanicData.csv'

	features, classes, x_train, y_train, x_test, y_test, data = load_data(path)

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

	print(f"\nHighest (abs) correlation with y (class): {max_corr[0]}  (feature '{features[indices[0]]}')")
	print(f"2nd highest (abs) correlation with y (class): {max_corr[1]}  (feature '{features[indices[1]]}')")

	w1, w2 = regressor.weights[indices]
	m = -w1 / w2
	c = -regressor.bias / w2
	x_scatter = np.append(x_train[:, indices], x_test[:, indices], axis=0)
	y_scatter = np.append(y_train, y_test)
	x_plot = np.array([np.min(x_scatter, axis=0)[0], np.max(x_scatter, axis=0)[0]])
	y_plot = m * x_plot + c

	for idx, class_label in enumerate(sorted(np.unique(y_scatter), reverse=True)):
		# This means that 'classes' will be indexed correctly with 'idx'
		plt.scatter(*x_scatter[y_scatter == class_label].T, alpha=0.7, label=classes[idx])
	plt.plot(x_plot, y_plot, color='black', ls='--')
	plt.ylim(np.min(x_scatter) * 1.1, np.max(x_scatter) * 1.1)
	plt.xlabel(features[indices[0]] + ' (standardised)')
	plt.ylabel(features[indices[1]] + ' (standardised)')
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
