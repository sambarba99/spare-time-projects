"""
Apollonian gasket drawing

Author: Sam Barba
Created 23/09/2022
"""

import tkinter as tk

import matplotlib.pyplot as plt

from generation import ApollonianGasketGenerator

plt.rcParams['figure.figsize'] = (7, 7)

def generate():
	r1 = slider_r1.get()
	r2 = slider_r2.get()
	r3 = slider_r3.get()
	generator = ApollonianGasketGenerator(r1, r2, r3)
	circles = generator.generate(slider_steps.get())  # Total circles after n steps = 2 * 3^n + 2

	plt.gca().cla()
	for circ in circles:
		plt_circle = plt.Circle((circ.centre.real, circ.centre.imag), circ.r, linewidth=0.5, fill=False)
		plt.gca().add_patch(plt_circle)
	plt.axis('scaled')
	plt.title(f'{len(circles)} circles')
	plt.show()

if __name__ == '__main__':
	root = tk.Tk()
	root.title('Apollonian Gasket Drawing')
	root.config(width=400, height=330, bg='#000024')
	root.eval('tk::PlaceWindow . center')
	root.resizable(False, False)

	select_iters_lbl = tk.Label(root, text='Select no. steps:', font='consolas', bg='#000024', fg='white')
	slider_steps = tk.Scale(root, from_=0, to=6, resolution=1, orient='horizontal', font='consolas', command=lambda _: generate())

	select_radii_lbl = tk.Label(root, text="Select 3 starting circles' radii:", font='consolas', bg='#000024', fg='white')
	slider_r1 = tk.Scale(root, from_=1, to=5, resolution=0.1, orient='horizontal', font='consolas',
		command=lambda _: generate())
	slider_r2 = tk.Scale(root, from_=1, to=5, resolution=0.1, orient='horizontal', font='consolas',
		command=lambda _: generate())
	slider_r3 = tk.Scale(root, from_=1, to=5, resolution=0.1, orient='horizontal', font='consolas',
		command=lambda _: generate())

	select_iters_lbl.place(relwidth=0.8, relheight=0.11, relx=0.5, rely=0.09, anchor='center')
	slider_steps.place(relwidth=0.6, relheight=0.14, relx=0.5, rely=0.22, anchor='center')
	select_radii_lbl.place(relwidth=0.8, relheight=0.11, relx=0.5, rely=0.37, anchor='center')
	slider_r1.place(relwidth=0.6, relheight=0.15, relx=0.5, rely=0.52, anchor='center')
	slider_r2.place(relwidth=0.6, relheight=0.15, relx=0.5, rely=0.69, anchor='center')
	slider_r3.place(relwidth=0.6, relheight=0.15, relx=0.5, rely=0.86, anchor='center')

	generate()

	root.mainloop()
