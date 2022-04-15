# Pearson Correlation Finder
# Author: Sam Barba
# Created 15/10/2021

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def partial_shuffle(data, percentage=10):
	indices = list(range(len(data)))
	n = round(len(data) * percentage / 100)
	indices = np.random.choice(indices, n, replace=False)
	mapping = dict((indices[i], indices[i - 1]) for i in range(n))
	return np.array([data[mapping.get(i, i)] for i in range(len(data))])

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
	choice = input("Enter R to generate random data, or E to enter it manually: ").upper()
	if choice == "R":
		x = np.random.uniform(-10, 10, size=200)
		y = np.random.uniform(-10, 10, size=200)
		x = partial_shuffle(sorted(x))
		y = partial_shuffle(sorted(y))
	else:
		x = np.array(list(map(float, input("\nInput the X sample: ").split())))
		y = np.array(list(map(float, input("Input the Y sample: ").split())))

	n = len(x)
	sum_x = x.sum()
	sum_y = y.sum()
	mean_x = sum_x / n
	mean_y = sum_y / n
	std_x = np.std(x, ddof=1)  # DDOF = 1 means sample standard deviation (not population)
	std_y = np.std(y, ddof=1)
	sum_x2 = (x * x).sum()
	sum_y2 = (y * y).sum()
	sum_xy = (x * y).sum()

	# Pearson's coefficient
	r_num = n * sum_xy - sum_x * sum_y
	r_denom1 = n * sum_x2 - sum_x ** 2
	r_denom2 = n * sum_y2 - sum_y ** 2
	r = r_num / ((r_denom1 * r_denom2) ** 0.5)

	# Regression line y = mx + c
	m = r * std_y / std_x
	c = mean_y - m * mean_x

	x_plot = [min(x), max(x)]
	y_plot = [m * i + c for i in x_plot]

	plt.figure(figsize=(8, 8))
	plt.scatter(x, y, color="black", alpha=0.4)
	plt.plot(x_plot, y_plot, color="red")
	if min(x) < 0 or min(y) < 0:
		plt.axhline(color="black")
		plt.axvline(color="black")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title(f"r = {r}\nY = {m:.3f}X {'+' if c >= 0 else '-'} {c:.3f}")
	plt.show()
