"""
Levenshtein distance demo

Author: Sam Barba
Created 15/03/2022
"""

import tkinter as tk

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def find_lev_dist(*_):
	print('\nLevenshtein distance table:\n')

	table = lev_dp(sv1.get(), sv2.get())

	# Add side header and top header
	for idx, c in enumerate(' ' + sv1.get()):
		table[idx].insert(0, c)
	table.insert(0, list(' ' + sv2.get()))

	for row in table:
		print(row)

	d = lev_recursive(sv1.get(), sv2.get())

	output_lbl.config(text=str(d))

def lev_dp(a, b):
	"""Dynamic programming implementation"""

	len_a, len_b = len(a), len(b)

	table = [[0] * (len_b + 1) for _ in range(len_a + 1)]

	for i in range(len_a + 1):
		table[i][0] = i
	for i in range(len_b + 1):
		table[0][i] = i

	for i in range(1, len_a + 1):
		for j in range(1, len_b + 1):
			if a[i - 1] == b[j - 1]:
				table[i][j] = table[i - 1][j - 1]
			else:
				table[i][j] = 1 + min(table[i - 1][j], table[i][j - 1], table[i - 1][j - 1])

	return table  # Distance = table[-1][-1]

def lev_recursive(a, b):
	"""Recursive implementation"""

	if len(a) == 0: return len(b)
	elif len(b) == 0: return len(a)
	elif a[0] == b[0]: return lev_recursive(a[1:], b[1:])
	else:
		return 1 + min(lev_recursive(a[1:], b), lev_recursive(a, b[1:]), lev_recursive(a[1:], b[1:]))

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	root = tk.Tk()
	root.title('Levenshtein distance calculator')
	root.config(width=350, height=200, bg='#000024')
	root.eval('tk::PlaceWindow . center')

	enter_strings_lbl = tk.Label(root, text='Enter 2 strings:', font='consolas', bg='#000024', fg='white')

	sv1 = tk.StringVar(value='python')
	sv2 = tk.StringVar(value='programming')
	sv1.trace_add(mode='write', callback=find_lev_dist)
	sv2.trace_add(mode='write', callback=find_lev_dist)

	entry_box1 = tk.Entry(root, textvariable=sv1, font='consolas', justify='center')
	entry_box2 = tk.Entry(root, textvariable=sv2, font='consolas', justify='center')

	result_lbl = tk.Label(root, text='Levenshtein distance:', font='consolas', bg='#000024', fg='white')
	output_lbl = tk.Label(root, font='consolas')

	enter_strings_lbl.place(relwidth=0.9, relheight=0.15, relx=0.5, rely=0.14, anchor='center')
	entry_box1.place(relwidth=0.8, relheight=0.15, relx=0.5, rely=0.3, anchor='center')
	entry_box2.place(relwidth=0.8, relheight=0.15, relx=0.5, rely=0.49, anchor='center')
	result_lbl.place(relwidth=0.9, relheight=0.1, relx=0.5, rely=0.67, anchor='center')
	output_lbl.place(relwidth=0.33, relheight=0.14, relx=0.5, rely=0.82, anchor='center')

	find_lev_dist()

	root.mainloop()
