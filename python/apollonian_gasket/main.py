"""
Apollonian Gasket fractal renderer

Author: Sam Barba
Created 23/09/2022
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from generation import ApollonianGasketGenerator


plt.rcParams['figure.figsize'] = (7, 7)


def generate(*_):
	num_steps = slider_depth.val
	r1 = slider_r1.val
	r2 = slider_r2.val
	r3 = slider_r3.val
	generator = ApollonianGasketGenerator(r1, r2, r3)
	circles = generator.generate(num_steps)  # Total circles after n steps = 2 * 3^n + 2

	ax_circles.clear()
	for circ in circles:
		plt_circle = plt.Circle((circ.centre.real, circ.centre.imag), circ.r, linewidth=0.5, fill=False)
		ax_circles.add_patch(plt_circle)
	ax_circles.axis('scaled')
	ax_circles.set_xticks([])
	ax_circles.set_yticks([])
	ax_circles.set_title(f'{len(circles)} circles')
	plt.show()


def reset(*_):
	slider_depth.reset()
	slider_r1.reset()
	slider_r2.reset()
	slider_r3.reset()
	generate()


if __name__ == '__main__':
	ax_circles = plt.axes([0.15, 0.22, 0.7, 0.7])
	ax_depth = plt.axes([0.22, 0.16, 0.45, 0.03])
	ax_r1 = plt.axes([0.22, 0.12, 0.45, 0.03])
	ax_r2 = plt.axes([0.22, 0.08, 0.45, 0.03])
	ax_r3 = plt.axes([0.22, 0.04, 0.45, 0.03])
	ax_reset = plt.axes([0.75, 0.09, 0.1, 0.05])

	slider_depth = Slider(ax_depth, 'Depth', 0, 6, valinit=0, valstep=1)
	slider_r1 = Slider(ax_r1, 'r1', 1, 5, valinit=1, valstep=0.1)
	slider_r2 = Slider(ax_r2, 'r2', 1, 5, valinit=1, valstep=0.1)
	slider_r3 = Slider(ax_r3, 'r3', 1, 5, valinit=1, valstep=0.1)
	reset_btn = Button(ax_reset, 'Reset')

	slider_depth.on_changed(generate)
	slider_r1.on_changed(generate)
	slider_r2.on_changed(generate)
	slider_r3.on_changed(generate)
	reset_btn.on_clicked(reset)

	generate()
