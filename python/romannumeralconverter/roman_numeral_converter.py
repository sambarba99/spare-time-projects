"""
Roman Numeral Converter

Author: Sam Barba
Created 06/09/2021
"""

from datetime import datetime
import tkinter as tk

NUMERAL_VALS = {'M': 1000,
	'CM': 900,
	'D': 500,
	'CD': 400,
	'C': 100,
	'XC': 90,
	'L': 50,
	'XL': 40,
	'X': 10,
	'IX': 9,
	'V': 5,
	'IV': 4,
	'I': 1}

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def convert(*_):
	input_s = sv.get().upper()

	try:
		int_s = abs(int(input_s))
	except Exception:
		int_s = None

	if input_s == '' or (int_s is None and len(set(input_s).difference(set('IVXLCDM')))):
		output_lbl.config(text='Bad input!')
		return

	if int_s is not None:
		output_lbl.config(text=f'{int_s} = {int_to_numerals(int_s)}')
	else:
		output_lbl.config(text=f'{input_s} = {numerals_to_int(input_s)}')

def int_to_numerals(n):
	if n == 0: return '0'

	numerals = ''
	for k, v in NUMERAL_VALS.items():
		while n >= v:
			numerals += k
			n -= v

	return numerals

def numerals_to_int(numerals):
	n = 0

	for idx, item in enumerate(numerals):
		val = NUMERAL_VALS[item]

		if idx + 1 < len(numerals) and NUMERAL_VALS[numerals[idx + 1]] > val:
			n -= val
		else:
			n += val

	return n

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	root = tk.Tk()
	root.title('Roman Numeral Converter')
	root.config(width=400, height=200, bg='#000024')
	root.eval('tk::PlaceWindow . center')

	enter_num_lbl = tk.Label(root, text='Enter a number or numerals:',
		font='consolas', bg='#000024', fg='white')

	sv = tk.StringVar(value=datetime.now().year)
	sv.trace_add(mode='write', callback=convert)
	entry_box = tk.Entry(root, textvariable=sv, font='consolas', justify='center')

	result_lbl = tk.Label(root, text='Result:', font='consolas', bg='#000024', fg='white')
	output_lbl = tk.Label(root, font='consolas')

	enter_num_lbl.place(relwidth=0.8, relheight=0.12, relx=0.5, rely=0.2, anchor='center')
	entry_box.place(relwidth=0.8, relheight=0.15, relx=0.5, rely=0.36, anchor='center')
	result_lbl.place(relwidth=0.8, relheight=0.12, relx=0.5, rely=0.56, anchor='center')
	output_lbl.place(relwidth=0.8, relheight=0.15, relx=0.5, rely=0.72, anchor='center')

	convert()

	root.mainloop()
