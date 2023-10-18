"""
Polynomial regression demo

Author: Sam Barba
Created 18/10/2023
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from polynomial_regressor import PolynomialRegressor


plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
np.random.seed(1)


if __name__ == '__main__':
	x_min, x_max = -1, 1
	noise = 1.2

	# Define an arbitrary cubic function (polynomial degree 3)

	x = np.linspace(x_min, x_max, 1000)
	y = 5 * x ** 3 - x ** 2 - 3 * x + 1  # 5x^3 - x^2 - 3x + 1
	y += np.random.uniform(-noise, noise, len(x))

	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

	# Sort data in order of x so it's plottable later

	train_idx = np.argsort(x_train)
	test_idx = np.argsort(x_test)
	x_train = x_train[train_idx]
	y_train = y_train[train_idx]
	x_test = x_test[test_idx]
	y_test = y_test[test_idx]

	# Apply correct shapes for model, and fit model

	x_train = x_train.reshape(-1, 1)
	x_test = x_test.reshape(-1, 1)
	y_train = y_train.reshape(-1, 1)
	y_test = y_test.reshape(-1, 1)

	poly_reg = PolynomialRegressor(degree=3)
	poly_reg.fit(x_train, y_train)

	# Print learned coefficients (descending order, i.e. x^3, x^2, ..., 1)

	print('Learned polynomial coefficients:', poly_reg.theta.flatten()[::-1])

	# Test model

	y_pred = poly_reg.predict(x_test)

	print('RMSE:', mean_squared_error(y_test, y_pred) ** 0.5)

	plt.plot(x_test, y_pred, color='red', label='Predicted test data')
	plt.scatter(x_test, y_test, color='blue', s=8, label='Actual test data')
	plt.legend()
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.show()
