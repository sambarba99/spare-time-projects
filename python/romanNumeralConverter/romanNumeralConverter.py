# Roman Numeral Converter
# Author: Sam Barba
# Created 06/09/2021

import tkinter as tk

ALL_NUMERAL_VALS = {"M": 1000,
	"CM": 900,
	"D": 500,
	"CD": 400,
	"C": 100,
	"XC": 90,
	"L": 50,
	"XL": 40,
	"X": 10,
	"IX": 9,
	"V": 5,
	"IV": 4,
	"I": 1}

SINGLE_NUMERAL_VALS = {"I": 1,
	"V": 5,
	"X": 10,
	"L": 50,
	"C": 100,
	"D": 500,
	"M": 1000}

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def int_to_numerals(n):
	if n <= 0: return

	global output_lbl

	numerals = ""
	for k, v in ALL_NUMERAL_VALS.items():
		while n >= v:
			numerals += k
			n -= v

	output_lbl.configure(text=numerals)

def numerals_to_int(numerals):
	global output_lbl

	n = 0

	for idx, item in enumerate(numerals):
		val = SINGLE_NUMERAL_VALS[item]

		if idx + 1 < len(numerals) and SINGLE_NUMERAL_VALS[numerals[idx + 1]] > val:
			n -= val
		else:
			n += val

	output_lbl.configure(text=str(n))

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

root = tk.Tk()
root.title("Roman Numeral Converter")
root.configure(width=400, height=270, bg="#141414")
root.eval("tk::PlaceWindow . center")

frame = tk.Frame(root, bg="#0080ff")
frame.place(relwidth=0.9, relheight=0.6, relx=0.5, rely=0.38, anchor="center")

enter_num_lbl = tk.Label(frame, text="Enter a number or numerals:", font="consolas", bg="#0080ff")
enter_num_lbl.place(relwidth=0.9, relheight=0.17, relx=0.5, rely=0.17, anchor="center")

entry_box = tk.Entry(frame, font="consolas", justify="center")
entry_box.place(relwidth=0.8, relheight=0.17, relx=0.5, rely=0.36, anchor="center")

btn_int_to_numerals = tk.Button(frame, text="Convert to\nnumerals", font="consolas", command=lambda: int_to_numerals(int(entry_box.get())))
btn_numerals_to_int = tk.Button(frame, text="Convert to\ninteger", font="consolas", command=lambda: numerals_to_int(entry_box.get().upper()))
btn_int_to_numerals.place(relwidth=0.38, relheight=0.35, relx=0.3, rely=0.73, anchor="center")
btn_numerals_to_int.place(relwidth=0.38, relheight=0.35, relx=0.7, rely=0.73, anchor="center")

result_lbl = tk.Label(root, text="Result:", font="consolas", bg="#141414", fg="#dcdcdc")
result_lbl.place(relwidth=0.9, relheight=0.1, relx=0.5, rely=0.76, anchor="center")

output_lbl = tk.Label(root, font="consolas")
output_lbl.place(relwidth=0.9, relheight=0.11, relx=0.5, rely=0.87, anchor="center")

root.mainloop()
