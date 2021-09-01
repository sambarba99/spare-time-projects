"""
Linear regression demo

Author: Sam Barba
Created 10/11/2021
"""

from linear_regressor import LinearRegressor
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (10, 7)

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

	# Standardise data (column-wise)
	split = int(len(data) * train_test_ratio)
	training_mean = np.mean(data[:split], axis=0)
	training_std = np.std(data[:split], axis=0)
	data = (data - training_mean) / training_std

	x, y = data[:, :-1], data[:, -1]
	x_train, y_train = x[:split], y[:split]
	x_test, y_test = x[split:], y[split:]

	return feature_names, x_train, y_train, x_test, y_test, data

def analytic_solution(x, y):
	# Adding dummy x0 = 1 makes the first weight w0 equal the bias
	x = [[1] + list(i) for i in list(x)]
	x = np.array(x)
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
		case 'B': path = 'C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\bostonData.txt'
		case 'C': path = 'C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\carValueData.txt'
		case _: path = 'C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\medicalInsuranceData.txt'

	feature_names, x_train, y_train, x_test, y_test, data = extract_data(path)

	weights, bias = analytic_solution(x_train, y_train)
	weights = ', '.join(f'{we:.3f}' for we in weights)
	print(f'\nAnalytic solution:\n weights = {weights}\n bias = {bias:.3f}\n')

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

	print('\nFeature names:', ', '.join(feature_names))
	print(f'Highest (abs) correlation with y ({feature_names[-1]}): {max_corr} '
		f"(feature '{feature_names[idx_max_corr]}')")

	weights = ', '.join(f'{we:.3f}' for we in regressor.weights)
	x_plot = np.append(x_train[:, idx_max_corr], x_test[:, idx_max_corr])
	y_plot = regressor.weights[idx_max_corr] * x_plot + regressor.bias
	y_scatter = np.append(y_train, y_test)
	plt.scatter(x_plot, y_scatter, color='black', alpha=0.6, s=10)
	plt.plot(x_plot, y_plot, color='red')
	plt.xlabel(feature_names[idx_max_corr] + ' (standardised)')
	plt.ylabel(feature_names[-1] + ' (standardised)')
	plt.title(f'Gradient descent solution\nweights = {weights}\nbias = {regressor.bias:.3f}')
	plt.show()

	# Plot MAE graph

	y_plot = np.array(regressor.cost_history)
	plt.plot(y_plot, color='red')
	plt.xlabel('Training iteration')
	plt.ylabel('MAE')
	plt.title('MAE during training')
	plt.show()

if __name__ == '__main__':
	main()
