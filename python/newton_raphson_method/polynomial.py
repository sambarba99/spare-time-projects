"""
Polynomial class for Newton-Raphson method demo

Author: Sam Barba
Created 14/10/2021
"""

import random


class Polynomial:
	def __init__(self, coefficients):
		"""
		Args:
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
		def format_number(n):
			s = f'{n:.4g}'
			if 'e' in s:
				base, exp = s.split('e')
				if base == '1':
					return fr'10^{{{int(exp)}}}'
				else:
					return fr'{base}\cdot10^{{{int(exp)}}}'  # E.g. 2.5 ⋅ 10^{-9} (LaTeX needs exponents wrapping in {})
			return s

		def power(degree):
			if degree == 0:
				return ''
			if degree == 1:
				return 'x'
			return f'x^{{{degree}}}'


		degree = len(self.coefficients) - 1
		terms = []

		for idx, c in enumerate(self.coefficients):
			if c == 0:
				continue

			d = degree - idx
			abs_c = abs(c)

			if d > 0 and abs_c == 1:
				coeff_str = ''
			else:
				if float(abs_c).is_integer():
					abs_c = int(abs_c)
				coeff_str = str(abs_c) if isinstance(abs_c, int) else format_number(abs_c)

			term = f'{coeff_str}{power(d)}'

			if not terms:
				terms.append(term if c > 0 else f'-{term}')
			else:
				sign = '+' if c > 0 else '-'
				terms.append(f'{sign}{term}')

		return ' '.join(terms)

	# Evaluate polynomial at x
	def __call__(self, x):
		ret = 0
		for idx, c in enumerate(self.coefficients[::-1]):
			ret += c * x ** idx
		return ret
