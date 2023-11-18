"""
Polynomial class for Newton-Raphson method demo

Author: Sam Barba
Created 14/10/2021
"""

import random


class Polynomial:
	def __init__(self, coefficients):
		"""
		Parameters:
			coefficients: Define the polynomial in the form a_n, a_(n-1), ..., a_0
		"""

		self.coefficients = coefficients

	def find_root(self, converge_threshold=1e-9, max_iters=1e6):
		"""Approximate solution of f(x) = 0 via Newton-Raphson method"""

		x0 = random.random()  # Initial guess

		df = self.derivative()
		xn = x0
		fxn = self(xn)

		i = 0
		for _ in range(int(max_iters)):
			dfxn = df(xn)
			if dfxn == 0:
				return 'Zero derivative: no solution'

			xn = xn - fxn / dfxn
			fxn = self(xn)

			if abs(fxn) < converge_threshold:
				break
			i += 1

		if abs(fxn) >= converge_threshold:
			return 'No solution found'

		return xn, i + 1, x0

	def derivative(self):
		deriv_coefficients = []
		exponent = len(self.coefficients) - 1
		for c in self.coefficients[:-1]:
			deriv_coefficients.append(c * exponent)
			exponent -= 1
		return Polynomial(deriv_coefficients)

	def __str__(self):
		def expr(degree):
			if degree == 0: return ''
			if degree == 1: return 'x'
			return f'x^{degree}'


		degree = len(self.coefficients) - 1
		result = ''

		for i in range(degree + 1):
			c = self.coefficients[i]

			if abs(c) == 1 and i < degree:
				result += ' +' if c > 0 else ' -'
				if i > 0: result += ' '
				result += expr(degree - i)
			elif c != 0:
				if c % 1 == 0: c = int(c)

				if c > 0: result += ' + '
				else: result += ' -' if i == 0 else ' - '

				result += f'{abs(c)}{expr(degree - i)}'

		return result.lstrip(' + ')  # Remove leading ' + '

	# Evaluate polynomial at x
	def __call__(self, x):
		result = 0
		for idx, c in enumerate(self.coefficients[::-1]):
			result += c * x ** idx
		return result
