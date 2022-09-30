"""
Number to words converter

Author: Sam Barba
Created 06/09/2022
"""

SMALL = {0: '', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
	10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen',
	17: 'seventeen', 18: 'eighteen', 19: 'nineteen'}

TENS = {2: 'twenty', 3: 'thirty', 4: 'forty', 5: 'fifty', 6: 'sixty', 7: 'seventy', 8: 'eighty', 9: 'ninety'}

BIG = {1: 'thousand', 2: 'million', 3: 'billion', 4: 'trillion', 5: 'quadrillion', 6: 'quintillion', 7: 'sextillion',
	8: 'septillion', 9: 'octillion', 10: 'nonillion', 11: 'decillion'}

import tkinter as tk

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def convert(*_):
	try:
		num = int(sv.get())
		output = num_to_word(num)
		output = output.replace(' - ', '-')  # 'twenty - one' -> 'twenty-one'
	except Exception:
		output = 'Bad entry! Must be int'

	output_txt.config(state='normal')
	output_txt.delete('1.0', tk.END)
	output_txt.insert('1.0', output)
	output_txt.tag_add('center', '1.0', tk.END)
	output_txt.config(state='disabled')

def num_to_word(n):
	def join_(*args):
		return ' '.join(filter(bool, args))

	def divide(dividend, divisor, magnitude):
		return join_(
			say_num_pos(dividend // divisor),
			magnitude,
			say_num_pos(dividend % divisor)
		)

	def say_num_pos(n):
		if n < 20:
			return SMALL[n]
		if n < 100:
			# These nums should be hyphenated (unless ending with 0), e.g. 21 -> 'twenty-one'
			last_digit = n % 10
			if last_digit == 0: return TENS[n // 10]
			return join_(TENS[n // 10], '-', SMALL[last_digit])
		if n < 1000:
			return divide(n, 100, 'hundred')

		illions_num = illions_name = ''
		for illions_num, illions_name in BIG.items():
			if n < 1000 ** (illions_num + 1):
				break

		return divide(n, 1000 ** illions_num, illions_name)

	if n < 0: return join_('minus', say_num_pos(-n))
	if n == 0: return 'zero'
	return say_num_pos(n)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	root = tk.Tk()
	root.title('Number to word converter')
	root.config(width=470, height=200, bg='#000024')
	root.eval('tk::PlaceWindow . center')
	root.resizable(False, False)

	enter_num_lbl = tk.Label(root, text='Enter number to convert:', font='consolas', bg='#000024', fg='white')

	sv = tk.StringVar(value='123456789')
	sv.trace_add(mode='write', callback=convert)

	num_entry = tk.Entry(root, textvariable=sv,font='consolas', justify='center')

	output_txt = tk.Text(root, bg='white', font='consolas', state='disabled')
	output_txt.tag_configure('center', justify='center')

	enter_num_lbl.place(relwidth=0.7, relheight=0.15, relx=0.5, rely=0.12, anchor='center')
	num_entry.place(relwidth=0.9, relheight=0.14, relx=0.5, rely=0.27, anchor='center')
	output_txt.place(relwidth=0.9, relheight=0.5, relx=0.5, rely=0.65, anchor='center')

	convert()

	root.mainloop()
