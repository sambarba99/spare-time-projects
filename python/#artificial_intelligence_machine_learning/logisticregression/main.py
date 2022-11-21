"""
Logistic regression using PCA

Author: Sam Barba
Created 10/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from logistic_regressor import LogisticRegressor

plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)

def load_data(path, train_test_ratio=0.8):
	df = pd.read_csv(path)
	print(f'\nRaw data:\n{df}')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]
	x_to_encode = x.select_dtypes(exclude=np.number).columns
	labels = sorted(y.unique())

	for col in x_to_encode:
		if len(x[col].unique()) > 2:
			one_hot = pd.get_dummies(x[col], prefix=col)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)
		else:  # Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True)

	y = pd.get_dummies(y, prefix='class', drop_first=True)

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}')

	data = pd.concat([x, y], axis=1).to_numpy().astype(float)
	x, y = data[:, :-1], data[:, -1].astype(int)

	pca = PCA(n_components=2)
	x = pca.fit_transform(x)

	# Standardise x
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio, stratify=y)
	training_mean = x_train.mean(axis=0)
	training_std = x_train.std(axis=0)
	x_train = (x_train - training_mean) / training_std
	x_test = (x_test - training_mean) / training_std

	return labels, x_train, y_train, x_test, y_test

def plot_confusion_matrix(actual, predictions, labels, is_training):
	cm = confusion_matrix(actual, predictions)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
	f1 = f1_score(actual, predictions)

	disp.plot(cmap=plt.cm.plasma)
	plt.title(f'{"Training" if is_training else "Test"} confusion matrix\n(F1 score: {f1})')
	plt.show()

if __name__ == '__main__':
	choice = input('\nEnter 1 to use banknote dataset,'
		+ '\nor 2 for breast tumour dataset\n>>> ')

	match choice:
		case '1': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\banknoteData.csv'
		case _: path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\breastTumourData.csv'

	labels, x_train, y_train, x_test, y_test = load_data(path)

	regressor = LogisticRegressor(labels)
	regressor.fit(x_train, y_train)

	# Plot confusion matrices

	train_pred = regressor.predict(x_train)
	test_pred = regressor.predict(x_test)
	plot_confusion_matrix(y_train, train_pred, labels, True)
	plot_confusion_matrix(y_test, test_pred, labels, False)

	# Plot cost history

	plt.plot(regressor.cost_history)
	plt.xlabel('Training iteration')
	plt.ylabel('Cost')
	plt.title('Cost during training')
	plt.show()
