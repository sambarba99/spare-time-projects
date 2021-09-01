"""
KMP algorithm demo

Author: Sam Barba
Created 11/09/2021
"""

import random
import tkinter as tk

TEXT_LEN = 1664

text = ''

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def generate_text():
	global text

	text = ''.join([random.choice(list('ABCDE')) for _ in range(TEXT_LEN)])
	output_txt.config(state='normal')
	output_txt.delete('1.0', tk.END)
	output_txt.insert('1.0', text)
	output_txt.tag_add('center', '1.0', tk.END)
	output_txt.config(state='disabled')

	kmp()

def kmp(*_):
	def populate_lps(pattern):
		lps = [0] * len(pattern)
		i, length = 1, 0  # Length of previous longest prefix suffix

		while i < len(pattern):
			if pattern[i] == pattern[length]:
				length += 1
				lps[i] = length
				i += 1
			else:
				if length != 0:
					length = lps[length - 1]
				else:
					lps[i] = 0
					i += 1

		return lps

	pattern = sv.get().upper()
	len_t, len_p = len(text), len(pattern)

	# Colour output all black, before highlighting matched positions in blue
	output_txt.tag_remove('fill_black', '1.0', tk.END)
	output_txt.tag_add('fill_black', '1.0', tk.END)
	output_txt.tag_config('fill_black', foreground='black')

	# Clear all previous (if any) highlight tags
	for i in range(TEXT_LEN):
		output_txt.tag_remove(f'highlight{i}', '1.0', tk.END)

	if len_p == 0:
		result_lbl.config(text='Pattern length must be > 0')
		return
	elif len_p > len_t:
		result_lbl.config(text='Pattern must not be longer than text')
		return

	# Create array for holding longest prefix suffix values for pattern
	lps = populate_lps(pattern)

	positions = []

	i = j = 0
	while i < len_t:
		if text[i] == pattern[j]:
			i += 1
			j += 1

		if j == len_p:
			positions.append(i - j)
			j = lps[j - 1]
		elif i < len_t and text[i] != pattern[j]:
			if j != 0:
				j = lps[j - 1]
			else:
				i += 1

	for pos in positions:
		output_txt.tag_add(f'highlight{pos}', f'1.{pos}', f'1.{pos + len(pattern)}')
		output_txt.tag_config(f'highlight{pos}', background='#20a020')

	result_lbl.config(text=f'{len(positions)} results found:')

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	root = tk.Tk()
	root.title('KMP demo')
	root.config(width=650, height=730, bg='#000024')
	root.eval('tk::PlaceWindow . center')

	button = tk.Button(root, text='Generate random text', font='consolas', command=lambda: generate_text())

	enter_pattern_lbl = tk.Label(root, text='Enter pattern to search:', font='consolas', bg='#000024', fg='white')

	sv = tk.StringVar(value='ABC')
	sv.trace_add(mode='write', callback=kmp)

	pattern_entry = tk.Entry(root, textvariable=sv,font='consolas', justify='center')

	result_lbl = tk.Label(root, font='consolas', bg='#000024', fg='white')
	output_txt = tk.Text(root, bg='white', font='consolas', state='disabled')
	output_txt.tag_configure('center', justify='center')

	button.place(relwidth=0.35, relheight=0.05, relx=0.5, rely=0.07, anchor='center')
	enter_pattern_lbl.place(relwidth=0.5, relheight=0.04, relx=0.5, rely=0.14, anchor='center')
	pattern_entry.place(relwidth=0.5, relheight=0.04, relx=0.5, rely=0.18, anchor='center')
	result_lbl.place(relwidth=0.8, relheight=0.04, relx=0.5, rely=0.24, anchor='center')
	output_txt.place(relwidth=0.9, relheight=0.69, relx=0.5, rely=0.61, anchor='center')

	generate_text()

	root.mainloop()
