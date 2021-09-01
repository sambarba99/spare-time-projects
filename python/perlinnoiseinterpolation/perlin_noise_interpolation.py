"""
Perlin noise demo

Author: Sam Barba
Created 17/05/2022
"""

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

plt.rcParams['figure.figsize'] = (8, 5)
plt.subplots_adjust(left=0.08, bottom=0.05, right=1.04, top=0.84)

slider = og_grid = plot_grid = None

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def generate_grid():
	global og_grid
	og_grid = np.random.uniform(0, 10, size=(80, 120))
	interpolate_and_plot()

def interpolate_and_plot():
	global plot_grid

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
	ax.set_title(f'Interpolation steps: {slider.get()}')
	plt.colorbar(mat, ax=ax)
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	root = tk.Tk()
	root.title('Perlin Noise Visualiser')
	root.config(width=320, height=180, bg='#000024')

	select_iters_lbl = tk.Label(root, text='Select no. steps:', font='consolas', bg='#000024', fg='white')

	slider = tk.Scale(root, from_=0, to=5, resolution=1, orient='horizontal', font='consolas',
		command=lambda _: interpolate_and_plot())

	btn_randomise_grid = tk.Button(root, text='Randomise grid', font='consolas',
		command=lambda: generate_grid())

	select_iters_lbl.place(relwidth=0.8, relheight=0.2, relx=0.5, rely=0.17, anchor='center')
	slider.place(relwidth=0.8, relheight=0.28, relx=0.5, rely=0.42, anchor='center')
	btn_randomise_grid.place(relwidth=0.8, relheight=0.2, relx=0.5, rely=0.75, anchor='center')

	generate_grid()

	root.mainloop()
