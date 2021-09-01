"""
Pi Calculator

Author: Sam Barba
Created 04/03/2019
"""

import matplotlib.pyplot as plt

N_DIGITS = 500

plt.rcParams['figure.figsize'] = (7, 5)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def calc_pi():
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

if __name__ == '__main__':
	pi = calc_pi()

	digits = [next(pi) for _ in range(N_DIGITS)]

	print(*digits, sep='')

	x_plot = range(10)
	y_plot = [0] * 10

	for idx, i in enumerate(digits):
		y_plot[i] += 1
		plt.cla()
		plt.bar(x_plot, y_plot, tick_label=x_plot, color='#0080ff')
		plt.xlabel('Digit')
		plt.ylabel('Occurrences')
		plt.title(f'Digit frequencies of {idx + 1} digits of pi')
		plt.show(block=False)
		plt.pause(1e-6)

	plt.show()
