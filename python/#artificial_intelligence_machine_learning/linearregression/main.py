"""
Linear regression demo on Boston housing dataset

Author: Sam Barba
Created 10/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from linear_regressor import LinearRegressor


plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)


def load_data(train_test_ratio=0.8):
	df = pd.read_csv(r'C:\Users\Sam\Desktop\Projects\datasets\boston_housing.csv')
	print(f'\nRaw data:\n{df}')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]
	x_to_encode = x.select_dtypes(exclude=np.number).columns
	y_name = df.columns[-1]

	for col in x_to_encode:
		n_unique = x[col].nunique()
		if n_unique == 1:
			# No information from this feature
			x = x.drop(col, axis=1)
		elif n_unique > 2:
			# Multivariate feature
			one_hot = pd.get_dummies(x[col], prefix=col)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)
		else:
			# Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True)
	features = x.columns

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}')

	data = pd.concat([x, y], axis=1).to_numpy()
	x, y = data[:, :-1], data[:, -1]

	# Keep only most correlated feature to y

	corr_coeffs = np.corrcoef(data.T)
	# Make bottom-right coefficient 0, as this doesn't count (correlation of last column with itself)
	corr_coeffs[-1, -1] = 0

	# Index of column with highest correlation with y
	idx = np.abs(corr_coeffs[:, -1]).argmax()
	# Keep strongest
	max_corr = corr_coeffs[idx, -1]
	feature = features[idx]
	x = x[:, idx]

	print(f"\nHighest (abs) correlation with y ({df.columns[-1]}): {max_corr}  (feature '{feature}')")

	# Standardise x if numeric, to aid gradient descent
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio, random_state=1)

	x_train = np.expand_dims(x_train, -1)
	y_train = np.expand_dims(y_train, -1)
	x_test = np.expand_dims(x_test, -1)
	y_test = np.expand_dims(y_test, -1)

	if feature not in x_to_encode:
		scaler = StandardScaler()
		x_train = scaler.fit_transform(x_train)
		x_test = scaler.transform(x_test)

	return x_train, y_train, x_test, y_test, feature, y_name


def ordinary_least_squares(x, y):
	# Adding dummy x0 = 1 makes the first weight w0 equal the bias
	x = np.hstack((np.ones((x.shape[0], 1)), x))
	solution = ((np.linalg.inv(x.T.dot(x))).dot(x.T)).dot(y)
	bias, weights = solution[0], solution[1:]
	return bias, weights


if __name__ == '__main__':
	x_train, y_train, x_test, y_test, feature, y_name = load_data()

	bias, weight = ordinary_least_squares(x_train, y_train)
	print(f'\nOLS solution: weight = {weight[0][0]:.3f}, bias = {bias[0]:.3f}')

	regressor = LinearRegressor(feature, y_name)
	regressor.fit(x_train, y_train)

	y_pred = regressor.predict(x_test)
	mae = np.abs(y_pred[:, 0] - y_test[:, 0]).sum() / len(y_test)
	print('\nTest MAE:', mae)

	# Plot cost history

	plt.plot(regressor.cost_history)
	plt.xlabel('Training iteration')
	plt.ylabel('MAE')
	plt.title('MAE during training')
	plt.show()
