# Pearson Correlation Finder
# Author: Sam Barba
# Created 15/10/2021

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def partialShuffle(data, percentage=10):
	indices = list(range(len(data)))
	n = round(len(data) * percentage / 100)
	indices = np.random.choice(indices, n, replace=False)
	mapping = dict((indices[i], indices[i - 1]) for i in range(n))
	return [data[mapping.get(i, i)] for i in range(len(data))]

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

choice = input("Enter R to generate random data, or E to enter it manually: ").upper()
if choice == "R":
	x = [np.random.uniform(-10, 10) for i in range(200)]
	y = [np.random.uniform(-10, 10) for i in range(200)]
	x = partialShuffle(sorted(x))
	y = partialShuffle(sorted(y))
else:
	x = list(map(float, input("\nInput the X sample: ").split()))
	y = list(map(float, input("Input the Y sample: ").split()))

n = len(x)
sumX = sum(x)
sumY = sum(y)
meanX = sumX / n
meanY = sumY / n
stdX = np.std(x, ddof=1) # DDOF = 1 means sample standard deviation (not population)
stdY = np.std(y, ddof=1)
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

xPlot = [min(x), max(x)]
yPlot = [m * i + c for i in xPlot]
plusMinus = "+" if c >= 0 else "-"

plt.figure(figsize=(8, 8))
plt.scatter(x, y, color="black", alpha=0.4)
plt.plot(xPlot, yPlot, color="red")
if min(x) < 0 or min(y) < 0:
	plt.axhline(color="black")
	plt.axvline(color="black")
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"r = {r}\nY = {round(m, 3)}X {plusMinus} {round(abs(c), 3)}")
plt.show()
