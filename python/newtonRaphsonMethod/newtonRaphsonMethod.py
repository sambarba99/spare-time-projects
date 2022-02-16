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
	solution = p.find_root()
else:
	solution = dp.find_root()

if solution is not None:
	root, iters, initial_guess = solution

	if abs(root) <= 0.01: # If root is close to 0
		start, end = -1, 1
	else:
		start, end = root - abs(root), root + abs(root)

	x_plot = list(np.linspace(start, end))
	y_plot = [p(x) for x in x_plot]
	y_deriv_plot = [dp(x) for x in x_plot]
	yMin = min(y_plot + y_deriv_plot)
	yMax = max(y_plot + y_deriv_plot)

	plt.figure(figsize=(8, 8))
	plt.plot(x_plot, y_plot, color="#0080ff")
	plt.plot(x_plot, y_deriv_plot, color="#ff8000")
	plt.axhline(color="black")
	plt.vlines(root, yMin, yMax, color="red", ls="--")
	plt.legend(["f(x)", "f'(x) = " + str(dp)])
	plt.xlabel("x")
	plt.ylabel("f(x) and f'(x)")
	if choice == "1":
		plt.title(f"f(x) = {str(p)}"
			+ f"\nRoot: x = {root}"
			+ f"\nFound after {iters} iterations (initial guess = {initial_guess})")
	else:
		rootY = round(p(root), 9)
		plt.title(f"f(x) = {str(p)}"
			+ f"\nStationary point: {root}, {rootY}"
			+ f"\nFound after {iters} iterations (initial guess = {initial_guess})")
	plt.show()
