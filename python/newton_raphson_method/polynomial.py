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
			i.e. f(x) = a_n(x^n) + a_(n-1)(x^(n-1)) + ... + a_1(x) + a_0
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

			xn -= fxn / dfxn
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

	def __repr__(self):
		def expr(degree):
			if degree == 0:
				return ''
			if degree == 1:
				return 'x'
			return f'x^{degree}'


		degree = len(self.coefficients) - 1
		ret = ''

		for i in range(degree + 1):
			c = self.coefficients[i]

			if abs(c) == 1 and i < degree:
				ret += ' +' if c > 0 else ' -'
				if i > 0:
					ret += ' '
				ret += expr(degree - i)
			elif c != 0:
				if c == (int_c := int(c)):
					c = int_c

				if c > 0:
					ret += ' + '
				else:
					ret += ' -' if i == 0 else ' - '

				ret += f'{abs(c)}{expr(degree - i)}'

		return ret.lstrip(' + ')

	# Evaluate polynomial at x
	def __call__(self, x):
		result = 0
		for idx, c in enumerate(self.coefficients[::-1]):
			result += c * x ** idx
		return result
