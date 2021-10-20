# Newton-Raphson method demo
# Author: Sam Barba
# Created 14/10/2021

import matplotlib.pyplot as plt
import numpy as np
from polynomial import Polynomial

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

coefficients = list(map(float, input("Input f(x) coefficients as shown below:"
	+ "\ne.g. for 2x^3 - 3x^2 + 5 enter: 2 -3 0 5\n").split()))

p = Polynomial(coefficients)
dp = p.derivative()

choice = input("\nEnter 1 to find a root of f(x), or 2 to find a stationary point of f(x): ")

if choice == "1":
	solution = p.findRoot()
else:
	solution = dp.findRoot()

if solution is not None:
	root, iters, initialGuess = solution

	if abs(root) <= 0.01: # If root is close to 0
		start, end = -1, 1
	else:
		start, end = root - abs(root), root + abs(root)

	xPlot = list(np.linspace(start, end))
	yPlot = [p(x) for x in xPlot]
	yDerivPlot = [dp(x) for x in xPlot]
	yMin = min(yPlot + yDerivPlot)
	yMax = max(yPlot + yDerivPlot)

	plt.figure(figsize=(8, 8))
	plt.plot(xPlot, yPlot, color="#0080ff")
	plt.plot(xPlot, yDerivPlot, color="#ff8000")
	plt.axhline(color="black")
	plt.vlines(root, yMin, yMax, color="red", ls="--")
	plt.legend(["f(x)", "f'(x) = " + str(dp)])
	plt.xlabel("x")
	plt.ylabel("f(x) and f'(x)")
	if choice == "1":
		plt.title(f"f(x) = {str(p)}"
			+ f"\nRoot: x = {root}"
			+ f"\nFound after {iters} iterations (initial guess = {initialGuess})")
	else:
		rootY = round(p(root), 9)
		plt.title(f"f(x) = {str(p)}"
			+ f"\nStationary point: {root}, {rootY}"
			+ f"\nFound after {iters} iterations (initial guess = {initialGuess})")
	plt.show()
