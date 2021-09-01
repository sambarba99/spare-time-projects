# Newton-Raphson method demo
# Author: Sam Barba
# Created 14/10/2021

import math
import matplotlib.pyplot as plt
import numpy as np
from polynomial import *
import random

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

while True:
	coefficients = list(map(float, input("Input f(x) coefficients as shown below:"
		+ "\ne.g. for 5x^3 + 2x - 1 enter: 5 0 2 -1\n").split()))

	p = Polynomial(coefficients)
	dp = p.derivative()

	print("\n f(x) =", str(p))
	print("f'(x) =", str(dp))
	root = p.findRoot(random.random())
	print("root =", root)
	print("f(root) =", p(root))

	if root != None:
		if abs(root) <= 0.01: # If root is close to 0
			start, end = -1, 1
		else:
			start, end = root - abs(root), root + abs(root)

		xPlot = list(np.linspace(start, end, 20))
		yPlot = [p(x) for x in xPlot]
		yDerivPlot = [dp(x) for x in xPlot]

		plt.plot(xPlot, yPlot, color = "#0080ff")
		plt.plot(xPlot, yDerivPlot, color = "#ff8000")
		plt.axhline(color = "red", linewidth = 1)
		plt.axvline(color = "red", linewidth = 1)
		plt.legend(["f(x) = " + str(p), "f'(x) = " + str(dp)])
		plt.xlabel("x")
		plt.ylabel("f(x)")
		plt.title("Newton-Raphson method demo")
		plt.show()

	choice = input("\nEnter to continue or X to exit: ").upper()
	if len(choice) > 0 and choice[0] == 'X':
		break
	print()
