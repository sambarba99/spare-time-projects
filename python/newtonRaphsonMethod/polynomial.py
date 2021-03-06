"""
Polynomial class for Newton-Raphson method demo

Author: Sam Barba
Created 14/10/2021
"""

import random

class Polynomial:
	def __init__(self, coefficients):
		# Coefficients are in the form a_n, a_(n-1) ... a_0
		self.coefficients = coefficients

	def find_root(self, converge_threshold=10 ** -9, max_iters=10 ** 5):
		"""Approximate solution of f(x) = 0 via Newton-Raphson method"""

		x0 = random.random()  # Initial guess

		df = self.derivative()
		xn = x0
		fxn = self(xn)

		i = 0
		for _ in range(max_iters):
			dfxn = df(xn)
			if dfxn == 0:
				return 'Zero derivative: No solution'

			xn = xn - fxn / dfxn
			fxn = self(xn)

			if abs(fxn) < converge_threshold:
				break
			i += 1

		if abs(fxn) >= converge_threshold:
			return 'No solution found'

		return xn, i + 1, x0

	def derivative(self):
		derived_coefficients = []
		exponent = len(self.coefficients) - 1
		for i in range(len(self.coefficients) - 1):
			derived_coefficients.append(self.coefficients[i] * exponent)
			exponent -= 1
		return Polynomial(derived_coefficients)

	def __str__(self):
		degree = len(self.coefficients) - 1
		result = ''

		for i in range(degree + 1):
			c = self.coefficients[i]

			if abs(c) == 1 and i < degree:
				result += (' +' if c > 0 else ' -')
				if i > 0: result += ' '
				result += self.__expr(degree - i)
			elif c != 0:
				if c % 1 == 0: c = int(c)

				if c > 0: result += ' + '
				else: result += ' -' if i == 0 else ' - '

				result += str(abs(c)) + self.__expr(degree - i)

		return result.lstrip(' + ')  # Remove leading ' + '

	def __expr(self, degree):
		if degree == 0: return ''
		if degree == 1: return 'x'
		return 'x^' + str(degree)

	# Evaluate polynomial at x
	def __call__(self, x):
		result = 0
		for idx, c in enumerate(self.coefficients[::-1]):
			result += c * x ** idx
		return result
