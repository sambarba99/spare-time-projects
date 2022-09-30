"""
Linear regression demo

Author: Sam Barba
Created 10/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from linear_regressor import LinearRegressor

plt.rcParams['figure.figsize'] = (10, 7)
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

	for col in x_to_encode:
		if len(x[col].unique()) > 2:
			one_hot = pd.get_dummies(x[col], prefix=col)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)
		else:  # Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True)
	features = x.columns

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}')

	data = pd.concat([x, y], axis=1).to_numpy().astype(float)
	x, y = data[:, :-1], data[:, -1]

	# Standardise x (numeric features only)
	numeric_feature_indices = [idx for idx, f in enumerate(features) if f not in x_to_encode]
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio)
	training_mean = np.mean(x_train[:, numeric_feature_indices], axis=0)
	training_std = np.std(x_train[:, numeric_feature_indices], axis=0)
	x_train[:, numeric_feature_indices] = (x_train[:, numeric_feature_indices] - training_mean) / training_std
	x_test[:, numeric_feature_indices] = (x_test[:, numeric_feature_indices] - training_mean) / training_std

	return features, x_train, y_train, x_test, y_test, data

def analytic_solution(x, y):
	# Adding dummy x0 = 1 makes the first weight w0 equal the bias
	x = np.hstack((np.ones((x.shape[0], 1)), x))
	solution = ((np.linalg.inv(x.T.dot(x))).dot(x.T)).dot(y)
	weights, bias = solution[1:], solution[0]
	return weights, bias

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	choice = input('Enter B to use Boston housing dataset,'
		+ '\nC for car value dataset,'
		+ '\nor M for medical insurance dataset\n>>> ').upper()

	match choice:
		case 'B': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\bostonData.csv'
		case 'C': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\carValueData.csv'
		case _: path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\medicalInsuranceData.csv'

	features, x_train, y_train, x_test, y_test, data = load_data(path)

	weights, bias = analytic_solution(x_train, y_train)
	weights = ', '.join(f'{we:.3f}' for we in weights)
	print(f'\nAnalytic solution:\nweights = {weights}\nbias = {bias:.3f}\n')

	regressor = LinearRegressor()
	regressor.fit(x_train, y_train)

	print('Training MAE:', regressor.cost_history[-1])
	print('Test MAE:', regressor.cost(x_test, y_test, regressor.weights, regressor.bias))

	# Plot regression line using column with the strongest correlation with y variable

	corr_coeffs = np.corrcoef(data.T)
	# Make bottom-right coefficient 0, as this doesn't count (correlation of last column with itself)
	corr_coeffs[-1, -1] = 0

	# Index of column that has the strongest correlation with y
	idx_max_corr = np.argmax(np.abs(corr_coeffs[:, -1]))
	max_corr = corr_coeffs[idx_max_corr, -1]

	print('\nFeatures:', ', '.join(features))
	print(f'Highest (abs) correlation with y ({features[-1]}): {max_corr} '
		f"(feature '{features[idx_max_corr]}')")

	weights = ', '.join(f'{we:.3f}' for we in regressor.weights)
	x_plot = np.append(x_train[:, idx_max_corr], x_test[:, idx_max_corr])
	y_plot = regressor.weights[idx_max_corr] * x_plot + regressor.bias
	y_scatter = np.append(y_train, y_test)
	plt.scatter(x_plot, y_scatter, color='black', alpha=0.6, s=10)
	plt.plot(x_plot, y_plot, color='red')
	plt.xlabel(features[idx_max_corr] + ' (standardised)')
	plt.ylabel(features[-1] + ' (standardised)')
	plt.title(f'Gradient descent solution\nweights = {weights}\nbias = {regressor.bias:.3f}')
	plt.show()

	# Plot MAE graph

	plt.plot(regressor.cost_history, color='red')
	plt.xlabel('Training iteration')
	plt.ylabel('MAE')
	plt.title('MAE during training')
	plt.show()

if __name__ == '__main__':
	main()
