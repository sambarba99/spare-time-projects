# Polynomial class for Newton-Raphson method demo
# Author: Sam Barba
# Created 14/10/2021

class Polynomial:
	def __init__(self, coefficients):
		# Coefficients are in the form a_n, a_(n-1) ... a_0
		self.coefficients = coefficients

	# Approximate solution of f(x) = 0 via Newton-Raphson method
	def findRoot(self, x0, tolerance = 10 ** -9, maxIter = 100000):
		df = self.derivative()
		xn = x0
		fxn = self(xn)
		iter = 0

		while abs(fxn) > tolerance and iter < maxIter:
			dfxn = df(xn)
			if dfxn == 0:
				print("\nZero derivative. No solution found - maybe try another initial guess?")
				return None
			xn = xn - fxn / dfxn
			fxn = self(xn)
			iter += 1

		print("\nFound root after {} iterations (initial guess = {})".format(iter, x0))
		return xn

	def derivative(self):
		derivedCoefficients = []
		exponent = len(self.coefficients) - 1
		for i in range(len(self.coefficients) - 1):
			derivedCoefficients.append(self.coefficients[i] * exponent)
			exponent -= 1
		return Polynomial(derivedCoefficients)

	def __str__(self):
		degree = len(self.coefficients) - 1
		result = ""

		for i in range(degree + 1):
			c = self.coefficients[i]

			if abs(c) == 1 and i < degree:
				result += (" +" if c > 0 else " -")
				if i > 0: result += " "
				result += Polynomial.__xExpr(degree - i)
			elif c != 0:
				# If c is int
				if c % 1 == 0: c = int(c)

				if c > 0: result += " + "
				else: result += " -" if i == 0 else " - "

				result += str(abs(c)) + Polynomial.__xExpr(degree - i)

		return result.lstrip(" + ") # Remove leading " + "

	def __xExpr(degree):
		if degree == 0: return ""
		if degree == 1: return "x"
		return "x^" + str(degree)

	# Evaluate polynomial at x
	def __call__(self, x):
		result = 0
		for idx, c in enumerate(self.coefficients[::-1]):
			result += c * x ** idx
		return result
