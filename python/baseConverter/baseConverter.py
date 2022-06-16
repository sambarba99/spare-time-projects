"""
Number Base Converter

Author: Sam Barba
Created 04/09/2021
"""

import tkinter as tk

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def convert(*_):
	global sv_num, sv_from_base, output_txt

	if len(sv_num.get()) == 0 or len(sv_from_base.get()) == 0:
		return

	num_str = sv_num.get()
	from_base = int(sv_from_base.get())

	input_num_to_dec = int(num_str) if from_base == 10 else to_decimal_from_base(num_str, from_base)

	to_other_bases = []

	for i in range(2, 17):
		if i == from_base: continue

		num_in_base_i = to_base_from_decimal(input_num_to_dec, i)
		to_other_bases.append(f'{num_str} from base {from_base} to base {i}: {num_in_base_i}')

	output_txt.config(state='normal')
	output_txt.delete('1.0', tk.END)
	output_txt.insert('1.0', '\n'.join(to_other_bases))
	output_txt.tag_add('center', '1.0', tk.END)
	output_txt.config(state='disabled')

def to_decimal_from_base(num_str, from_base):
	dec_num = 0
	power = from_base ** (len(num_str) - 1)

	for n in num_str:
		val = ord(n) - ord('0') if '0' <= n <= '9' else ord(n) - ord('a') + 10
		dec_num += val * power
		power //= from_base

	return dec_num

def to_base_from_decimal(dec_num, to_base):
	remainders = []

	while dec_num:
		remainder = dec_num % to_base
		remainder = str(remainder) if remainder < 10 else str(chr(87 + remainder))
		remainders.append(remainder)
		dec_num //= to_base

	return ''.join(remainders[::-1])

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	root = tk.Tk()
	root.title('Base Converter')
	root.config(width=650, height=480, bg='#000045')
	root.eval('tk::PlaceWindow . center')

	enter_num_lbl = tk.Label(root, text='Enter a number:', font='consolas', bg='#000045', fg='white')
	enter_base_lbl = tk.Label(root, text='Enter its base:', font='consolas', bg='#000045', fg='white')
	enter_num_lbl.place(relwidth=0.5, relheight=0.05, relx=0.5, rely=0.08, anchor='center')
	enter_base_lbl.place(relwidth=0.5, relheight=0.05, relx=0.5, rely=0.21, anchor='center')

	sv_num = tk.StringVar(value='2000')
	sv_from_base = tk.StringVar(value='10')
	sv_num.trace_add(mode='write', callback=convert)
	sv_from_base.trace_add(mode='write', callback=convert)

	num_entry = tk.Entry(root, textvariable=sv_num, font='consolas', justify='center')
	base_entry = tk.Entry(root, textvariable=sv_from_base,font='consolas', justify='center')
	num_entry.place(relwidth=0.3, relheight=0.06, relx=0.5, rely=0.14, anchor='center')
	base_entry.place(relwidth=0.3, relheight=0.06, relx=0.5, rely=0.27, anchor='center')

	output_txt = tk.Text(root, bg='white', font='consolas', state='disabled')
	output_txt.tag_configure('center', justify='center')
	output_txt.place(relwidth=0.9, relheight=0.57, relx=0.5, rely=0.63, anchor='center')

	convert()

	root.mainloop()
