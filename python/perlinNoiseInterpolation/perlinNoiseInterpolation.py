# Perlin noise demo
# Author: Sam Barba
# Created 17/05/2022

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

plt.rcParams["figure.figsize"] = (10, 6)

og_grid = plot_grid = slider = ax = None

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def generate_grid():
	global og_grid
	og_grid = np.random.uniform(0, 10, size=(80, 120))
	interpolate_and_plot()

def interpolate_and_plot():
	global slider, og_grid, plot_grid, ax

	rows, cols = og_grid.shape
	plot_grid = og_grid.copy()
	temp_copy = plot_grid.copy()

	for _ in range(slider.get()):
		for (y, x), _ in np.ndenumerate(plot_grid):
			sub_arr = plot_grid[max(0, y - 1):min(rows, y + 2), max(0, x - 1):min(cols, x + 2)]
			temp_copy[y][x] = np.mean(sub_arr)
		plot_grid = temp_copy.copy()

	ax = plt.subplot()
	mat = ax.matshow(plot_grid)
	ax.set_title(f"Interpolation steps: {slider.get()}")
	plt.colorbar(mat, ax=ax)
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
	root = tk.Tk()
	root.title("Perlin Noise Visualiser")
	root.config(width=320, height=180, bg="#141414")

	frame = tk.Frame(root, bg="#0080ff")
	frame.place(relwidth=0.9, relheight=0.9, relx=0.5, rely=0.5, anchor="center")

	select_iters_lbl = tk.Label(frame, text="Select no. steps:", font="consolas", bg="#0080ff")
	select_iters_lbl.place(relwidth=0.8, relheight=0.2, relx=0.5, rely=0.17, anchor="center")

	slider = tk.Scale(frame, from_=0, to=5, resolution=1, orient="horizontal", font="consolas",
		command=lambda _: interpolate_and_plot())
	slider.place(relwidth=0.8, relheight=0.28, relx=0.5, rely=0.42, anchor="center")

	btn_randomise_grid = tk.Button(frame, text="Randomise grid", font="consolas",
		command=lambda: generate_grid())
	btn_randomise_grid.place(relwidth=0.8, relheight=0.25, relx=0.5, rely=0.75, anchor="center")

	generate_grid()

	root.mainloop()
