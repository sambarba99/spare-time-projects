"""
Matrix class for matrix_calculator.py

Author: Sam Barba
Created 03/09/2021
"""

from math import radians, sin, cos

class Matrix:
	def __init__(self, values, rows=None, cols=None):
		def create_grid_from_line(values):
			float_list = list(map(float, values.split()))
			grid = [[0] * self.cols for _ in range(self.rows)]

			for idx, item in enumerate(float_list):
				row, col = idx // self.cols, idx % self.cols
				grid[row][col] = item

			return grid

		self.rows = rows if rows else len(values)
		self.cols = cols if cols else len(values[0])
		self.grid = values if isinstance(values, list) else create_grid_from_line(values)

	def add_subtract(self, other, add):
		if add:
			result_grid = [[self.grid[i][j] + other.grid[i][j] for j in range(self.cols)] for i in range(self.rows)]
		else:
			result_grid = [[self.grid[i][j] - other.grid[i][j] for j in range(self.cols)] for i in range(self.rows)]

		return Matrix(result_grid)

	def mult(self, other):
		result_grid = [[0] * other.cols for _ in range(self.rows)]

		for i in range(self.rows):
			for j in range(other.cols):
				result_grid[i][j] = sum(self.grid[i][k] * other.grid[k][j] for k in range(self.cols))

		return Matrix(result_grid)

	def determinant(self, rows):
		match rows:
			case 1: return self.grid[0][0]
			case 2: return self.grid[0][0] * self.grid[1][1] - self.grid[0][1] * self.grid[1][0]

		det = 0
		sub_grid = [[0] * (rows - 1) for _ in range(rows - 1)]

		for i in range(rows):
			for j in range(rows):
				col = 0
				for k in range(rows):
					if k != i:
						sub_grid[j - 1][col] = self.grid[j][k]
						col += 1
			plus_minus_one = 1 if i % 2 == 0 else -1
			submatrix = Matrix(sub_grid)
			det += self.grid[0][i] * plus_minus_one * submatrix.determinant(rows - 1)

		return det

	def inverse(self):
		"""
		Inverse of a square matrix = 1/determinant * adjugate matrix
		= 1/determinant * transposed cofactor matrix
		"""

		det = self.determinant(self.rows)
		comatrix = self.comatrix()
		adjugate_grid = list(zip(*comatrix.grid))  # Transposed comatrix
		result_grid = [[x / det for x in row] for row in adjugate_grid]

		return Matrix(result_grid)

	def comatrix(self):
		def remove_row_and_col(mat, row, col):
			sub_grid = [[0] * (mat.cols - 1) for _ in range(mat.rows - 1)]
			sub_row = sub_col = 0

			for i in range(mat.rows):
				if i == row: continue

				for j in range(mat.cols):
					if j == col: continue

					sub_grid[sub_row][sub_col] = mat.grid[i][j]
					sub_col = (sub_col + 1) % len(sub_grid)
					if sub_col == 0:
						sub_row += 1

			return Matrix(sub_grid)

		result_grid = [[0] * self.cols for _ in range(self.rows)]

		for i in range(self.rows):
			for j in range(self.cols):
				submatrix = remove_row_and_col(self, i, j)
				plus_minus_one = 1 if (i + j) % 2 == 0 else -1
				result_grid[i][j] = submatrix.determinant(submatrix.rows) * plus_minus_one

		return Matrix(result_grid)

	def power(self, p):
		r = Matrix(self.grid)
		if p < 0:
			r = r.inverse()
			p = -p

		result = Matrix(r.grid)

		while p > 1:
			result = result.mult(r)
			p -= 1

		return result

	def rref(self):
		rref_grid = self.grid

		pivot_col = 0

		for row in range(self.rows):
			# 1. Find left-most nonzero entry (pivot)
			pivot_row = row
			while rref_grid[pivot_row][pivot_col] == 0:
				pivot_row += 1
				if pivot_row == self.rows:
					# Go back to initial row, but move to next column
					pivot_row = row
					pivot_col += 1
					if pivot_col == self.cols:
						# No pivot
						return Matrix(rref_grid)

			# 2. If pivot row is below current row, swap these rows (so 0s are pushed to bottom)
			if pivot_row > row:
				rref_grid[pivot_row], rref_grid[row] = rref_grid[row], rref_grid[pivot_row]

			# 3. Scale current row such that pivot becomes 1
			scale = rref_grid[row][pivot_col]
			if scale != 1:
				rref_grid[row] = [x / scale for x in rref_grid[row]]

			# 4. Make entries above/below equal 0
			for r in range(self.rows):
				scale = rref_grid[r][pivot_col]
				if r != row and scale != 0:
					rref_grid[r] = [x - scale * y for x, y in zip(rref_grid[r], rref_grid[row])]

			# 5. Move to next column
			pivot_col += 1
			if pivot_col == self.cols:
				return Matrix(rref_grid)

		return Matrix(rref_grid)

	# ---------------------------------- Geometric transformations ---------------------------------- #

	def translate(self, dx, dy):
		result_grid = self.grid

		for i in range(self.rows):
			result_grid[i][0] += dx
			result_grid[i][1] += dy

		return Matrix(result_grid)

	def enlarge(self, k, x, y):
		"""Enlarge by factor k about (x, y)"""

		t = Matrix(self.grid)
		if x != 0 or y != 0:
			t = t.translate(-x, -y)  # In order to enlarge from origin (0, 0)

		enlarge_matrix = Matrix([[k, 0], [0, k]])
		result = t.mult(enlarge_matrix)

		# Undo first translation if necessary
		return result.translate(x, y) if x != 0 or y != 0 else result

	def reflect(self, m, c):
		"""Reflect across line y = mx + c"""

		t = Matrix(self.grid)
		if c != 0:
			t = t.translate(0, -c)  # Reflect in y = mx

		r = 1 / (1 + m ** 2)
		reflect_grid = [[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]
		reflect_matrix = Matrix([[x * r for x in row] for row in reflect_grid])

		result = t.mult(reflect_matrix)

		return result.translate(0, c) if c != 0 else result

	def rotate(self, theta, x, y):
		"""Rotate by theta (deg) clockwise about (x, y)"""

		t = Matrix(self.grid)
		if x != 0 or y != 0:
			t = t.translate(-x, -y)

		theta = radians(theta)
		sin_theta, cos_theta = sin(theta), cos(theta)
		rotate_matrix = Matrix([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

		result = t.mult(rotate_matrix)

		# Undo first translation if necessary
		return result.translate(x, y) if x != 0 or y != 0 else result

	def __repr__(self):
		return '\n'.join([' '.join([str(int(x)) if x % 1 == 0 else str(x) for x in row]) for row in self.grid])
