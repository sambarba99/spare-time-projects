"""
Root/stationary point finder using Newton-Raphson method

Author: Sam Barba
Created 14/10/2021
"""

import matplotlib.pyplot as plt
import numpy as np
from polynomial import Polynomial
import tkinter as tk

plt.rcParams['figure.figsize'] = (10, 6)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def handle_button_click(*, coefficients, find_root):
	# If find_root = True, find a root of f(x). Else, find a stationary point of f(x)
	# (i.e. a root of the derivative of f(x)).

	try:
		coefficients = [float(i) for i in coefficients.split()]
	except:
		entry_box.delete(0, tk.END)
		entry_box.insert(0, 'Bad input!')
		return

	polynomial = Polynomial(coefficients)
	poly_deriv = polynomial.derivative()

	solution = polynomial.find_root() if find_root else poly_deriv.find_root()

	if isinstance(solution, str):  # No solution
		entry_box.delete(0, tk.END)
		entry_box.insert(0, solution)
		return

	root, iters, initial_guess = solution

	if abs(root) <= 0.01:  # If root is close to 0
		start, end = -1, 1
	else:
		start, end = root - abs(root), root + abs(root)

	x_plot = np.linspace(start, end)
	y_plot = [polynomial(x) for x in x_plot]
	y_deriv_plot = [poly_deriv(x) for x in x_plot]
	y_min = min(y_plot + y_deriv_plot)
	y_max = max(y_plot + y_deriv_plot)

	plt.cla()
	plt.plot(x_plot, y_plot, color='#0080ff', label='f(x)')
	plt.plot(x_plot, y_deriv_plot, color='#ff8000', label=f"f'(x) = {str(poly_deriv)}")
	plt.axhline(color='black')
	plt.vlines(root, y_min, y_max, color='red', ls='--', linewidth=1)
	plt.xlabel('x')
	plt.ylabel("f(x) and f'(x)")
	plt.legend()

	if find_root:
		plt.title(f'f(x) = {str(polynomial)}'
			+ f'\nRoot: x = {root:.6f}'
			+ f'\nFound after {iters} iterations (initial guess = {initial_guess:.6f})')
	else:
		stationary_y = polynomial(root)
		plt.title(f'f(x) = {str(polynomial)}'
			+ f'\nStationary point: {root:.6f}, {stationary_y:.6f}'
			+ f'\nFound after {iters} iterations (initial guess = {initial_guess:.6f})')
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	root = tk.Tk()
	root.title('Root/Stationary Point Finder')
	root.config(width=400, height=200, bg='#000024')
	root.eval('tk::PlaceWindow . center')
	root.resizable(False, False)

	enter_coeffs_lbl = tk.Label(root, text='Enter polynomial coefficients\n(e.g. for 3x^2 - 5 enter: 3 0 -5):',
		font='consolas', bg='#000024', fg='white')

	entry_box = tk.Entry(root, font='consolas', justify='center')

	btn_find_root = tk.Button(root, text='Find root', font='consolas',
		command=lambda: handle_button_click(coefficients=entry_box.get(), find_root=True))
	btn_find_stat_point = tk.Button(root, text='Find stationary point', font='consolas',
		command=lambda: handle_button_click(coefficients=entry_box.get(), find_root=False))

	enter_coeffs_lbl.place(relwidth=0.9, relheight=0.2, relx=0.5, rely=0.17, anchor='center')
	entry_box.place(relwidth=0.8, relheight=0.15, relx=0.5, rely=0.4, anchor='center')
	btn_find_root.place(relwidth=0.58, relheight=0.17, relx=0.5, rely=0.64, anchor='center')
	btn_find_stat_point.place(relwidth=0.58, relheight=0.17, relx=0.5, rely=0.83, anchor='center')

	root.mainloop()
