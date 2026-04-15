"""
Functions for polynomial interpolation and evaluation

Author: Sam Barba
Created 15/04/2026
"""

import numpy as np


def interpolate(points):
	"""
	Compute coefficients a_n, a_(n-1), ..., a_0
	such that f(x) = a_n(x^n) + a_(n-1)(x^(n-1)) + ... + a_1(x) + a_0
	passes through ``points``.

	``points`` is an iterable of distinct (x,y) pairs. This function constructs the Vandermonde matrix corresponding to
	the x values, augments it with the y values, and solves the resulting linear system via Gauss-Jordan elimination (by
	reducing the matrix to reduced row echelon form, RREF).

	1. The Vandermonde matrix `V` is defined as `V[i, j] = x_i^(n - j - 1)`
	2. We solve the system `V * a = y` for the coefficient vector `a`
	3. The solution is obtained by converting the augmented matrix `[V | y]` into RREF and reading off the final column.

	See:
	- https://en.wikipedia.org/wiki/Vandermonde_matrix)
	- https://en.wikipedia.org/wiki/Row_echelon_form#Reduced_row_echelon_form)
	"""

	# Construct augmented Vandermonde matrix

	x, y = zip(*points)
	n = len(points)
	v = np.vander(x, n)
	v_augmented = np.column_stack((v, y))

	# Solve linear system via Gauss-Jordan elimination

	rows, cols = v_augmented.shape
	i = 0  # Current row
	pivot_rows, pivot_cols = [], []

	for pivot_col in range(cols - 1):  # Exclude the augmented column (y)
		# Find pivot (largest absolute value in the current column, from row `i` downwards)
		pivot_row = max(range(i, rows), key=lambda r: abs(v_augmented[r, pivot_col]))

		# If pivot is ~0, skip this col
		if abs(v_augmented[pivot_row, pivot_col]) < 1e-12:
			continue

		# Move the best pivot into the current row by swapping if necessary
		if pivot_row != i:
			v_augmented[[pivot_row, i]] = v_augmented[[i, pivot_row]]

		# Scale current row such that pivot becomes 1
		scale = v_augmented[i, pivot_col]
		v_augmented[i] /= scale

		# Make entries above/below current row equal 0 (eliminate)
		for r in range(rows):
			if r != i:
				factor = v_augmented[r, pivot_col]
				v_augmented[r] -= factor * v_augmented[i]

		# Keep track of these for coefficient extraction later
		pivot_rows.append(i)
		pivot_cols.append(pivot_col)

		# Move to next row
		i += 1
		if i >= rows:
			break

	# Extract resulting coefficients
	coeffs = np.zeros(cols - 1, dtype=np.float32)
	for row, col in zip(pivot_rows, pivot_cols):
		coeffs[col] = v_augmented[row, -1]

	return coeffs


def evaluate(coeffs, x):
	ret = 0
	for idx, c in enumerate(coeffs[::-1]):
		ret += c * x ** idx
	return ret


def polynomial_to_string(coeffs):
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


	degree = len(coeffs) - 1
	terms = []

	for idx, c in enumerate(coeffs):
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

	return fr"$f(x) = {' '.join(terms)}$"
