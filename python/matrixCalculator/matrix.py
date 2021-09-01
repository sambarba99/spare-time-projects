# Matrix class
# Author: Sam Barba
# Created 03/09/2021

import math

class Matrix:
	def __init__(self, values, rows = None, cols = None):
		self.rows = rows if rows != None else len(values)
		self.cols = cols if cols != None else len(values[0])
		self.grid = values if isinstance(values, list) else self.__createGridFromLine(values)

	def __createGridFromLine(self, values):
		floatList = list(map(float, values.split()))
		grid = [[0] * self.cols for i in range(self.rows)]

		for i in range(len(floatList)):
			row, col = i // self.cols, i % self.cols
			grid[row][col] = floatList[i]

		return grid

	def addSubtract(self, other, add):
		if add:
			resultGrid = [[self.grid[i][j] + other.grid[i][j] for j in range(self.cols)] for i in range(self.rows)]
		else:
			resultGrid = [[self.grid[i][j] - other.grid[i][j] for j in range(self.cols)] for i in range(self.rows)]

		return Matrix(resultGrid)

	def mult(self, other):
		resultGrid = [[0] * other.cols for i in range(self.rows)]

		for i in range(self.rows):
			for j in range(other.cols):
				for k in range(self.cols):
					resultGrid[i][j] += self.grid[i][k] * other.grid[k][j]

		return Matrix(resultGrid)

	def determinant(self, rows):
		if rows == 1:
			return self.grid[0][0]
		elif rows == 2:
			return self.grid[0][0] * self.grid[1][1] - self.grid[0][1] * self.grid[1][0]

		det = 0
		subGrid = [[0] * (rows - 1) for i in range(rows - 1)]

		for i in range(rows):
			for j in range(rows):
				col = 0
				for k in range(rows):
					if k != i:
						subGrid[j - 1][col] = self.grid[j][k]
						col += 1
			plusMinusOne = 1 if i % 2 == 0 else -1
			submatrix = Matrix(subGrid)
			det += self.grid[0][i] * plusMinusOne * submatrix.determinant(rows - 1)

		return det

	def inverse(self):
		det = self.determinant(self.rows)
		com = self.comatrix()
		com.grid = [[x / det for x in row] for row in com.grid]

		# Then transpose comatrix
		transposedComatrixGrid = [[com.grid[j][i] for j in range(self.rows)] for i in range(self.rows)]
		return Matrix(transposedComatrixGrid)

	def comatrix(self):
		resultGrid = [[0] * self.cols for i in range(self.rows)]

		for i in range(self.rows):
			for j in range(self.cols):
				submatrix = self.__removeRowAndCol(i, j)
				plusMinusOne = 1 if (i + j) % 2 == 0 else -1
				resultGrid[i][j] = submatrix.determinant(submatrix.rows) * plusMinusOne

		return Matrix(resultGrid)

	def __removeRowAndCol(self, row, col):
		subGrid = [[0]* (self.cols - 1) for i in range(self.rows - 1)]
		subRow = subCol = 0

		for i in range(self.rows):
			if i == row: continue

			for j in range(self.cols):
				if j == col: continue

				subGrid[subRow][subCol] = self.grid[i][j]
				subCol = (subCol + 1) % len(subGrid)
				if subCol == 0:
					subRow += 1

		return Matrix(subGrid)

	def power(self, p):
		if p < 0:
			self = self.inverse()
			p = -p
	
		result = Matrix(self.grid)

		while p > 1:
			result = result.mult(self)
			p -= 1

		return result

	def rref(self):
		rrefGrid = self.grid

		pivotRow = pivotCol = 0

		for row in range(self.rows):
			# 1. Find left-most nonzero entry (pivot)
			pivotRow = row
			while rrefGrid[pivotRow][pivotCol] == 0:
				pivotRow += 1
				if pivotRow == self.rows:
					# Go back to initial row, but move to next column
					pivotRow = row
					pivotCol += 1
					if pivotCol == self.cols:
						# No pivot
						return Matrix(rrefGrid)

			# 2. If pivot row is below current row, swap these rows (so 0s are pushed to bottom)
			if pivotRow > row:
				rrefGrid[pivotRow], rrefGrid[row] = rrefGrid[row], rrefGrid[pivotRow]

			# 3. Scale current row such that pivot becomes 1
			scale = rrefGrid[row][pivotCol]
			if scale != 1:
				rrefGrid[row] = [x / scale for x in rrefGrid[row]]

			# 4. Make entries above/below equal 0
			for r in range(self.rows):
				scale = rrefGrid[r][pivotCol]
				if r != row and scale != 0:
					rrefGrid[r] = [x - scale * y for x, y in zip(rrefGrid[r], rrefGrid[row])]

			# 5. Move to next column
			pivotCol += 1
			if pivotCol == self.cols:
				return Matrix(rrefGrid)

		return Matrix(rrefGrid)

	# ---------------------------------- Geometric transformations ---------------------------------- #

	def translate(self, dx, dy):
		resultGrid = self.grid

		for i in range(self.rows):
			resultGrid[i][0] += dx
			resultGrid[i][1] += dy

		return Matrix(resultGrid)

	def enlarge(self, k, x, y): # enlarge by factor k about (x, y)
		if x != 0 or y != 0:
			self = self.translate(-x, -y) # in order to enlarge from origin (0, 0)

		enlargeMatrix = Matrix([[k, 0], [0, k]])
		result = self.mult(enlargeMatrix)
	
		# undo first translation if necessary
		return result.translate(x, y) if x != 0 or y != 0 else result

	def reflect(self, m, c): # reflect across line y = mx + c
		if c != 0:
			self = self.translate(0, -c) # reflect in y = mx

		r = 1 / (1 + m ** 2)
		reflectGrid = [[1 - m ** 2, 2 * m], [2 * m, m ** 2 - 1]]
		reflectMatrix = Matrix([[x * r for x in row] for row in reflectGrid])

		result = self.mult(reflectMatrix)

		return result.translate(0, c) if c != 0 else result

	def rotate(self, a, x, y): # rotate by a° clockwise about (x, y)
		if x != 0 or y != 0:
			self = self.translate(-x, -y) # enlarge from origin (0, 0)

		a = math.radians(a)
		rotateMatrix = Matrix([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

		result = self.mult(rotateMatrix)

		# undo first translation if necessary
		return result.translate(x, y) if x != 0 or y != 0 else result

	def __repr__(self):
		return "\n".join([" ".join([str(int(x)) if x % 1 == 0 else str(x) for x in row]) for row in self.grid])
