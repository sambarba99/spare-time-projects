# Number Base Converter
# Author: Sam Barba
# Created 04/09/2021

import tkinter as tk

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def convert(num_str, from_base):
	global output_txt

	input_num_to_dec = int(num_str) if from_base == 10 else to_decimal_from_base(num_str, from_base)

	to_other_bases = []

	for i in range(2, 17):
		if i == from_base: continue

		num_in_base_i = to_base_from_decimal(input_num_to_dec, i)
		to_other_bases.append(f"{num_str} from base {from_base} to base {i}: {num_in_base_i}")

	output_txt.configure(state="normal")
	output_txt.delete("1.0", tk.END)
	output_txt.insert("1.0", "\n".join(to_other_bases))
	output_txt.tag_add("center", "1.0", tk.END)
	output_txt.configure(state="disabled")

def to_decimal_from_base(num_str, from_base):
	dec_num = 0
	power = from_base ** (len(num_str) - 1)

	for n in num_str:
		val = ord(n) - ord("0") if "0" <= n <= "9" else ord(n) - ord("a") + 10
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

	return "".join(remainders[::-1])

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

root = tk.Tk()
root.title("Base Converter")
root.configure(width=650, height=480, bg="#141414")
root.eval("tk::PlaceWindow . center")

upper_frame = tk.Frame(root, bg="#0080ff")
upper_frame.place(relwidth=0.75, relheight=0.23, relx=0.5, rely=0.18, anchor="center")

enter_num_lbl = tk.Label(upper_frame, text="Enter a number:", font="consolas", bg="#0080ff")
enter_base_lbl = tk.Label(upper_frame, text="Enter its base:", font="consolas", bg="#0080ff")
enter_num_lbl.place(relwidth=0.5, relheight=0.27, relx=0.2, rely=0.28, anchor="center")
enter_base_lbl.place(relwidth=0.5, relheight=0.27, relx=0.2, rely=0.72, anchor="center")

num_entry = tk.Entry(upper_frame, font="consolas", justify="center")
base_entry = tk.Entry(upper_frame, font="consolas", justify="center")
num_entry.place(relwidth=0.3, relheight=0.27, relx=0.5, rely=0.28, anchor="center")
base_entry.place(relwidth=0.3, relheight=0.27, relx=0.5, rely=0.72, anchor="center")

button = tk.Button(upper_frame, text="Convert to\nother bases", font="consolas", command=lambda: convert(num_entry.get().lower(), int(base_entry.get())))
button.place(relwidth=0.25, relheight=0.5, relx=0.81, rely=0.5, anchor="center")

output_txt = tk.Text(root, bg="#dcdcdc", font="consolas", state="disabled")
output_txt.tag_configure("center", justify="center")
output_txt.place(relwidth=0.9, relheight=0.57, relx=0.5, rely=0.65, anchor="center")

root.mainloop()
