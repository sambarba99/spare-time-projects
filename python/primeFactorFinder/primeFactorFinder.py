"""
Prime factor finder

Author: Sam Barba
Created 06/04/2022
"""

import tkinter as tk

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def find_factors(*_):
	global sv, output_lbl

	n, bad_input = None, False

	try:
		n = int(sv.get())
		bad_input = n <= 1
	except Exception:
		bad_input = True

	if bad_input:
		output_lbl.config(text='Enter an integer > 1')
		return

	pf = dict()  # Prime factors and their exponents
	while n % 2 == 0:
		if 2 in pf: pf[2] += 1
		else: pf[2] = 1
		n //= 2

	i = 3
	while n > 1:
		if n % i == 0:
			if i in pf: pf[i] += 1
			else: pf[i] = 1
			n //= i
		else:
			i += 2

	pf_str = ' x '.join(f'{prime}^{exp}' if exp > 1 else f'{prime}' for prime, exp in pf.items())

	output_lbl.config(text=pf_str)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	root = tk.Tk()
	root.title('Prime factor finder')
	root.config(width=340, height=180, bg='#000045')
	root.eval('tk::PlaceWindow . center')

	enter_num_lbl = tk.Label(root, text='Enter an integer:', font='consolas', bg='#000045', fg='white')
	enter_num_lbl.place(relwidth=0.8, relheight=0.12, relx=0.5, rely=0.2, anchor='center')

	sv = tk.StringVar(value='123456789')
	sv.trace_add(mode='write', callback=find_factors)
	entry_box = tk.Entry(root, textvariable=sv, font='consolas', justify='center')
	entry_box.place(relwidth=0.6, relheight=0.15, relx=0.5, rely=0.36, anchor='center')

	result_lbl = tk.Label(root, text='Prime factorisation:', font='consolas', bg='#000045', fg='white')
	result_lbl.place(relwidth=0.8, relheight=0.12, relx=0.5, rely=0.57, anchor='center')

	output_lbl = tk.Label(root, font='consolas', bg='#000045', fg='white')
	output_lbl.place(relwidth=0.8, relheight=0.12, relx=0.5, rely=0.75, anchor='center')

	find_factors()

	root.mainloop()
