# Root/stationary point finder using Newton-Raphson method
# Author: Sam Barba
# Created 14/10/2021

import matplotlib.pyplot as plt
import numpy as np
from polynomial import Polynomial
import tkinter as tk

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def handle_button_click(coefficients, find_root):
	global entry_box

	try:
		coefficients = [float(i) for i in coefficients.split()]
	except:
		entry_box.delete(0, tk.END)
		entry_box.insert(0, "Bad input!")
		return

	p = Polynomial(coefficients)
	dp = p.derivative()

	solution = p.find_root() if find_root else dp.find_root()

	if isinstance(solution, str):
		entry_box.delete(0, tk.END)
		entry_box.insert(0, solution)
		return

	if solution is not None:
		root, iters, initial_guess = solution

		if abs(root) <= 0.01: # If root is close to 0
			start, end = -1, 1
		else:
			start, end = root - abs(root), root + abs(root)

		x_plot = list(np.linspace(start, end))
		y_plot = [p(x) for x in x_plot]
		y_deriv_plot = [dp(x) for x in x_plot]
		yMin = min(y_plot + y_deriv_plot)
		yMax = max(y_plot + y_deriv_plot)

		plt.figure(figsize=(6, 6))
		plt.plot(x_plot, y_plot, color="#0080ff")
		plt.plot(x_plot, y_deriv_plot, color="#ff8000")
		plt.axhline(color="black")
		plt.vlines(root, yMin, yMax, color="red", ls="--")
		plt.legend(["f(x)", "f'(x) = " + str(dp)])
		plt.xlabel("x")
		plt.ylabel("f(x) and f'(x)")

		if find_root:
			plt.title(f"f(x) = {str(p)}"
				+ f"\nRoot: x = {root:.9f}"
				+ f"\nFound after {iters} iterations (initial guess = {initial_guess})")
		else:
			rootY = p(root)
			plt.title(f"f(x) = {str(p)}"
				+ f"\nStationary point: {root:.9f}, {rootY:.9f}"
				+ f"\nFound after {iters} iterations (initial guess = {initial_guess})")
		plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

root = tk.Tk()
root.title("Root/Stationary Point Finder")
root.configure(width=400, height=200, bg="#141414")
root.eval("tk::PlaceWindow . center")

frame = tk.Frame(root, bg="#0080ff")
frame.place(relwidth=0.9, relheight=0.9, relx=0.5, rely=0.5, anchor="center")

enter_coeffs_lbl = tk.Label(frame, text="Enter polynomial coefficients\n(e.g. for 3x^2 - 5 enter: 3 0 -5):", font="consolas", bg="#0080ff")
enter_coeffs_lbl.place(relwidth=0.9, relheight=0.25, relx=0.5, rely=0.17, anchor="center")

entry_box = tk.Entry(frame, font="consolas", justify="center")
entry_box.place(relwidth=0.8, relheight=0.15, relx=0.5, rely=0.4, anchor="center")

btn_find_root = tk.Button(frame, text="Find root", font="consolas", command=lambda: handle_button_click(entry_box.get(), True))
btn_find_stat_point = tk.Button(frame, text="Find stationary point", font="consolas", command=lambda: handle_button_click(entry_box.get(), False))
btn_find_root.place(relwidth=0.58, relheight=0.17, relx=0.5, rely=0.64, anchor="center")
btn_find_stat_point.place(relwidth=0.58, relheight=0.17, relx=0.5, rely=0.83, anchor="center")

root.mainloop()
