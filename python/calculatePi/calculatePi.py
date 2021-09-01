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
			n = ((10 * (3 * b + a)) // c) - 10 * n # // = division without remainder
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

digits = [next(pi) for i in range(size)]

digitFreq = {i: 0 for i in range(10)}

for n in digits:
	digitFreq[n] += 1

print("".join(str(d) for d in digits))

left = [i for i in range(10)]
height = digitFreq.values()
tickLabel = left[:]

plt.bar(left, height, tick_label = tickLabel, width = 0.8, color = "#0080ff")
plt.xlabel("Digit")
plt.ylabel("Occurrences")
plt.title("Digit frequencies in {} digits of pi".format(size))
plt.show()
