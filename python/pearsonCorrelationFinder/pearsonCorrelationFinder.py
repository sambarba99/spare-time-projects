# Pearson Correlation Finder
# Author: Sam Barba
# Created 15/10/2021

import matplotlib.pyplot as plt
import numpy as np
import random

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def partialShuffle(data, percentage = 5):
	n = round(len(data) * percentage / 100)
	indices = list(range(len(data)))
	random.shuffle(indices)
	indices = indices[:n]
	mapping = dict((indices[i], indices[i - 1]) for i in range(n))
	return [data[mapping.get(i, i)] for i in range(len(data))]

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

while True:
	choice = input("Enter R to generate random data, or E to enter it manually: ").upper()
	if choice == "R":
		x = [random.random() * random.randint(-10, 10) for i in range(200)]
		y = [random.random() * random.randint(-10, 10) for i in range(200)]
		x.sort()
		y.sort()
		x = partialShuffle(x)
		y = partialShuffle(y)
	else:
		x = list(map(float, input("\nInput the X sample: ").split()))
		y = list(map(float, input("Input the Y sample: ").split()))

	n = len(x)
	sumX = sum(x)
	sumY = sum(y)
	meanX = sumX / n
	meanY = sumY / n
	stdX = np.std(x, ddof = 1) # DDOF = 1 means sample standard deviation (not population)
	stdY = np.std(y, ddof = 1)
	sumX2 = sum(i * i for i in x)
	sumY2 = sum(j * j for j in y)
	sumXY = sum(i * j for i, j in zip(x, y))

	# Pearson's coefficient
	rNum = n * sumXY - sumX * sumY
	rDenom1 = n * sumX2 - sumX ** 2
	rDenom2 = n * sumY2 - sumY ** 2
	r = rNum / ((rDenom1 * rDenom2) ** 0.5)

	# Regression line y = mx + c
	m = r * stdY / stdX
	c = meanY - m * meanX

	xRegLine = [min(x), max(x)]
	yRegLine = [m * i + c for i in xRegLine]
	plusMinus = "+" if c >= 0 else "-"

	plt.scatter(x, y, color = "black", s = 2)
	plt.plot(xRegLine, yRegLine, color = "#0080ff", linewidth = 1)
	if min(x) < 0 or min(y) < 0:
		plt.axhline(color = "red", linewidth = 1)
		plt.axvline(color = "red", linewidth = 1)
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title("r = {}\nY = {}X {} {}".format(r, round(m, 3), plusMinus, round(abs(c), 3)))
	plt.show()

	choice = input("\nEnter to continue or X to exit: ").upper()
	if len(choice) > 0 and choice[0] == 'X':
		break
	print()
