"""
SciPy demo

Author: Sam Barba
Created 27/10/2022
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

plt.rcParams['figure.figsize'] = (9, 6)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

def optimisation():
	"""
	Minimise f(x,y) = (x-1)^2 + (y-2.5)^2 subject to:

	- x - 2y + 2 >= 0
	- -x - 2y + 6 >= 0
	- -x + 2y + 2 >= 0
	- x, y >= 0
	"""

	f = lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2
	constraints = (
		{'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
		{'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
		{'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2}
	)
	bounds = ((0, None), (0, None))
	res = scipy.optimize.minimize(f, x0=(2, 0), bounds=bounds, constraints=constraints)
	print(res)

def interpolation():
	x = np.linspace(0, 10, 10)
	y = x ** 2 * np.sin(x)
	x_dense = np.linspace(0, 10, 100)

	f_linear = scipy.interpolate.interp1d(x, y)
	f_cubic = scipy.interpolate.interp1d(x, y, kind='cubic')
	y_dense_linear = f_linear(x_dense)
	y_dense_cubic = f_cubic(x_dense)

	plt.scatter(x, y, color='black')
	plt.plot(x_dense, y_dense_linear, label='linear')
	plt.plot(x_dense, y_dense_cubic, label='cubic')
	plt.legend()
	plt.title('Interpolation')
	plt.show()

def derivation():
	f = lambda x: x ** 2 * np.sin(3 * x) * np.exp(-x)
	x = np.linspace(0, 1, 100)

	plt.plot(x, f(x), label='f(x)')
	plt.plot(x, scipy.misc.derivative(f, x, 1e-6), label="f'(x)")
	plt.plot(x, scipy.misc.derivative(f, x, 1e-6, n=2), label="f''(x)")
	plt.legend()
	plt.title('Derivation')
	plt.show()

def integration():
	f = lambda x: np.sin(x ** 0.5) * np.exp(-x)
	x = np.linspace(0, 2 * np.pi, 100)
	y = f(x)

	res = scipy.integrate.quad(f, 0, 2 * np.pi)
	plt.plot(x, y)
	plt.title(f'Integration\nArea = {res[0]}, error = {res[1]}')
	plt.fill_between(x, y, color='tab:blue', alpha=0.2)
	plt.show()

def curve_fitting():
	"""
	The equation for spring motion is y(t) = Acos(wt + phi). We want to find the natural frequency
	of oscillation (w) for a spring.
	"""

	# The following lab data is collected, to be used to determine A, w, phi:

	t_data = np.array([0.0000, 0.3448, 0.6897, 1.0345, 1.3793, 1.7241, 2.0690, 2.4138, 2.7586,
		3.1034, 3.4483, 3.7931, 4.1379, 4.4828, 4.8276, 5.1724, 5.5172, 5.8621, 6.2069, 6.5517,
		6.8966, 7.2414, 7.5862, 7.9310, 8.2759, 8.6207, 8.9655, 9.3103, 9.6552, 10.0000])
	y_data = np.array([4.3303, 1.6114, -2.1542, -3.9014, -1.6726, 2.1688, 3.8664, 1.8519, -1.8489,
		-3.9656, -2.1339, 1.5943, 4.0615, 1.8930, -1.7687, -4.2679, -2.4687, 1.3702, 4.2495, 2.2704,
		-1.5030, -3.4677, -2.5085, 1.2002, 3.8163, 2.9151, -1.2457, -3.7272, -2.5455, 0.8726])

	plt.plot(t_data, y_data, 'o--')
	plt.xlabel(r'$t$')
	plt.ylabel(r'$y$')
	plt.title('Sampled data')
	plt.show()

	f = lambda t, A, w, phi: A * np.cos(w * t + phi)

	# Initial guesses:

	A = 4
	w = np.pi
	phi = 0

	curve_fit = scipy.optimize.curve_fit(f, t_data, y_data, p0=(A, w, phi))
	optimal_params = curve_fit[0]
	param_covariance = curve_fit[1]
	A_opt, w_opt, phi_opt = optimal_params
	A_error, w_error, phi_error = np.diag(param_covariance) ** 0.5

	# Plot function with optimal found params
	t = np.linspace(0, 10, 500)
	y = f(t, A_opt, w_opt, phi_opt)
	plt.plot(t, y)
	plt.xlabel(r'$t$')
	plt.ylabel(r'$y$')
	plt.title(f'Fitted curve\n'
		fr'$A, \omega, \phi$ = {A_opt:.4f}, {w_opt:.4f}, {phi_opt:.4f}'
		f'\nErrors = {A_error:.4f}, {w_error:.4f}, {phi_error:.4f}')
	plt.show()

if __name__ == '__main__':
	print('----- Optimisation -----')
	optimisation()
	print('\n----- Interpolation -----')
	interpolation()
	print('\n----- Derivation -----')
	derivation()
	print('\n----- Integration -----')
	integration()
	print('\n----- Curve fitting -----')
	curve_fitting()
