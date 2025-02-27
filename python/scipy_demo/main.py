"""
SciPy demo

Author: Sam Barba
Created 27/10/2022
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy


plt.rcParams['figure.figsize'] = (6, 4)
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

	spline = scipy.interpolate.UnivariateSpline(x, f(x), s=0)
	deriv1 = spline.derivative()
	deriv2 = spline.derivative(n=2)

	plt.plot(x, f(x), label='f(x)')
	plt.plot(x, deriv1(x), label="f'(x)")
	plt.plot(x, deriv2(x), label="f''(x)")
	plt.legend()
	plt.title('Derivation')
	plt.show()


def integration():
	f = lambda x: np.sin(x ** 0.5) * np.exp(-x)
	x = np.linspace(0, 2 * np.pi, 100)
	y = f(x)

	res = scipy.integrate.quad(f, 0, 2 * np.pi)
	plt.plot(x, y)
	plt.title(f'Integration\nArea = {res[0]:.3f}, error = {res[1]:.3f}')
	plt.fill_between(x, y, color='tab:blue', alpha=0.2)
	plt.show()


def curve_fitting():
	"""
	The equation for spring motion is y(t) = Acos(wt + phi). We want to find the natural frequency
	of oscillation (w) for a spring.
	"""

	# The following lab data is collected, to be used to determine A, w, phi:

	t_data = np.array([0.00, 0.34, 0.69, 1.03, 1.38, 1.72, 2.07, 2.41, 2.76, 3.10, 3.45, 3.79, 4.14, 4.48, 4.83, 5.17,
		5.52, 5.86, 6.21, 6.55, 6.90, 7.24, 7.59, 7.93, 8.28, 8.62, 8.97, 9.31, 9.66, 10.00])
	y_data = np.array([4.33, 1.61, -2.15, -3.90, -1.67, 2.17, 3.87, 1.85, -1.85, -3.97, -2.13, 1.59, 4.06, 1.89, -1.77,
		-4.27, -2.47, 1.37, 4.25, 2.27, -1.50, -3.47, -2.51, 1.20, 3.82, 2.92, -1.25, -3.73, -2.55, 0.87])

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
		fr'$A, \omega, \phi$ = {A_opt:.3f}, {w_opt:.3f}, {phi_opt:.3f}'
		f'\nErrors = {A_error:.3f}, {w_error:.3f}, {phi_error:.3f}')
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
