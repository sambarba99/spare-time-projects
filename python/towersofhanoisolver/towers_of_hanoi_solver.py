"""
Towers of Hanoi solver

Author: Sam Barba
Created 20/09/2021
"""

import tkinter as tk

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def solve_and_write_steps():
	def solve(n, t1=1, t2=2, t3=3):
		if n:
			solve(n - 1, t1, t3, t2)
			steps.append(f'{len(steps) + 1}: Move disc from {t1} to {t3}')
			solve(n - 1, t2, t1, t3)

	steps = []

	n_discs = slider.get()
	solve(n_discs)

	steps_lbl.config(text=f'Steps ({len(steps)}):')

	output_steps.config(state='normal')
	output_steps.delete('1.0', tk.END)
	output_steps.insert('1.0', '\n'.join(steps))
	output_steps.tag_add('center', '1.0', tk.END)
	output_steps.config(state='disabled')

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	root = tk.Tk()
	root.title('Towers of Hanoi solver')
	root.config(width=500, height=650, bg='#000024')
	root.eval('tk::PlaceWindow . center')

	select_ratio_lbl = tk.Label(root, text='Select no. discs:', font='consolas', bg='#000024', fg='white')

	slider = tk.Scale(root, from_=1, to=12, orient='horizontal', font='consolas',
		command=lambda l: solve_and_write_steps())

	steps_lbl = tk.Label(root, font='consolas', bg='#000024', fg='white')

	output_steps = tk.Text(root, bg='white', font='consolas', state='disabled')
	output_steps.tag_configure('center', justify='center')

	select_ratio_lbl.place(relwidth=0.8, relheight=0.05, relx=0.5, rely=0.07, anchor='center')
	slider.place(relwidth=0.8, relheight=0.09, relx=0.5, rely=0.16, anchor='center')
	steps_lbl.place(relwidth=0.8, relheight=0.05, relx=0.5, rely=0.25, anchor='center')
	output_steps.place(relwidth=0.8, relheight=0.65, relx=0.5, rely=0.61, anchor='center')

	solve_and_write_steps()

	root.mainloop()
