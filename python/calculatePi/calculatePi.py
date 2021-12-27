# Pi Calculator
# Author: Sam Barba
# Created 04/03/2019

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def calcPi():
	a, b, c, d, e, n = 0, 1, 1, 1, 3, 3

	while True:
		if 4 * b + a - c < n * c:
			yield n
			na = 10 * (a - n * c)
			n = ((10 * (3 * b + a)) // c) - 10 * n
			b *= 10
			a = na
		else:
			na = (2 * b + a) * e
			nn = (b * (7 * d) + 2 + (a * e)) // (c * e)
			b *= d
			c *= e
			e += 2
			d += 1
			n, a = nn, na

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

size = 10 ** 3

pi = calcPi()

digits = [next(pi) for _ in range(size)]

print(*digits, sep="")

xPlot = list(range(10))
yPlot = [digits.count(i) for i in xPlot]

plt.figure(figsize=(8, 6))
plt.bar(xPlot, yPlot, tick_label=xPlot, color="#0080ff")
plt.xlabel("Digit")
plt.ylabel("Occurrences")
plt.title(f"Digit frequencies in {size} digits of pi")
plt.show()
