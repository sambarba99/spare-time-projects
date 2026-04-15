"""
Interactive Polynomial Interpolation

Author: Sam Barba
Created 15/04/2026
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import numpy as np

from polynomial import interpolate, evaluate, polynomial_to_string


plt.rcParams['figure.figsize'] = (9, 6)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

points = set()


def add_point(*_):
	"""Add a point, and update the interpolated polynomial"""

	try:
		x, y = map(float, textbox.text.split(','))
	except:
		print('Bad format: should be x,y')
		return

	points.add((x, y))
	print('Points:', sorted(points))
	textbox.set_val('')
	compute_polynomial_and_plot()


def reset(*_):
	global points

	points = set()
	textbox.set_val('')
	compute_polynomial_and_plot()


def compute_polynomial_and_plot():
	global points

	ax_plot.clear()
	ax_plot.set_title('Num. points < 2')
	if points:
		x, y = zip(*points)
		if len(set(x)) < len(x):
			print('x values must be unique')
			points = set()
		else:
			if len(points) >= 2:
				x_vals = np.linspace(min(x), max(x), 500)
				if len(set(y)) == 1:
					# Constant function e.g. f(x) = 5, f(x) = -1.2
					if y[0].is_integer():
						polynomial_str = fr'$f(x) = {int(y[0])}$'
					else:
						polynomial_str = fr'$f(x) = {y[0]}$'
					y_vals = np.repeat(y[0], len(x_vals))
				else:
					coeffs = interpolate(points)
					polynomial_str = polynomial_to_string(coeffs)
					y_vals = [evaluate(coeffs, xi) for xi in x_vals]
				ax_plot.plot(x_vals, y_vals, color='red', linestyle='--', linewidth=1, zorder=2)
				ax_plot.set_title(polynomial_str)
			ax_plot.scatter(x, y, color='red', zorder=2)
			ax_plot.axhline(color='black', linewidth=1, zorder=1)
			ax_plot.axvline(color='black', linewidth=1, zorder=1)
	ax_plot.set_xlabel('$x$', size=12)
	ax_plot.set_ylabel('$y$', size=12)
	plt.show()


if __name__ == '__main__':
	ax_plot = plt.axes([0.12, 0.25, 0.8, 0.66])
	ax_textbox = plt.axes([0.374, 0.08, 0.08, 0.05])
	ax_add_button = plt.axes([0.46, 0.08, 0.12, 0.05])
	ax_reset_button = plt.axes([0.585, 0.08, 0.12, 0.05])

	textbox = TextBox(ax_textbox, 'Enter coords: ', textalignment='center')
	add_point_btn = Button(ax_add_button, 'Add point')
	reset_btn = Button(ax_reset_button, 'Reset')
	add_point_btn.on_clicked(add_point)
	reset_btn.on_clicked(reset)

	compute_polynomial_and_plot()
