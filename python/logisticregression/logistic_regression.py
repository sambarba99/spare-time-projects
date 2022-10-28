"""
Logistic regression demo

Author: Sam Barba
Created 10/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from logistic_regressor import LogisticRegressor

plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
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
	classes = sorted(y.unique())

	for col in x_to_encode:
		if len(x[col].unique()) > 2:
			one_hot = pd.get_dummies(x[col], prefix=col)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)
		else:  # Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True)
	features = x.columns

	y = pd.get_dummies(y, prefix='class', drop_first=True)

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}')

	data = pd.concat([x, y], axis=1).to_numpy().astype(float)
	x, y = data[:, :-1], data[:, -1].astype(int)

	# Keep only 2 most correlated features to y

	corr_coeffs = np.corrcoef(data.T)
	# Make bottom-right coefficient 0, as this doesn't count (correlation of last column with itself)
	corr_coeffs[-1, -1] = 0

	# Indices of columns in descending order of correlation with y
	indices = np.abs(corr_coeffs[:, -1]).argsort()[::-1]
	# Keep 2 strongest
	indices = indices[:2]
	max_corr = corr_coeffs[indices, -1]
	features = features[indices]
	x = x[:, indices]

	print(f"\nHighest (abs) correlation with y (class): {max_corr[0]}  (feature '{features[0]}')")
	print(f"2nd highest (abs) correlation with y (class): {max_corr[1]}  (feature '{features[1]}')")

	# Standardise x
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio, stratify=y)
	training_mean = x_train.mean(axis=0)
	training_std = x_train.std(axis=0)
	x_train = (x_train - training_mean) / training_std
	x_test = (x_test - training_mean) / training_std

	return features, classes, x_train, y_train, x_test, y_test

def confusion_matrix(predictions, actual):
	n_classes = len(np.unique(actual))
	conf_mat = np.zeros((n_classes, n_classes)).astype(int)

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

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	choice = input('\nEnter 1 to use banknote dataset,'
		+ '\nor 2 for breast tumour dataset\n>>> ')

	match choice:
		case '1': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\banknoteData.csv'
		case _: path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\breastTumourData.csv'

	features, classes, x_train, y_train, x_test, y_test = load_data(path)

	regressor = LogisticRegressor(features, classes)
	regressor.fit(x_train, y_train)

	# Plot confusion matrices

	train_pred = regressor.predict(x_train)
	test_pred = regressor.predict(x_test)
	train_conf_mat, train_f1 = confusion_matrix(train_pred, y_train)
	test_conf_mat, test_f1 = confusion_matrix(test_pred, y_test)

	plot_confusion_matrices(train_conf_mat, train_f1, test_conf_mat, test_f1)

	# Plot cost history

	plt.plot(regressor.cost_history)
	plt.xlabel('Training iteration')
	plt.ylabel('Cost')
	plt.title('Cost during training')
	plt.show()

if __name__ == '__main__':
	main()
