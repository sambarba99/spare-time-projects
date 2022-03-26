# KMP algorithm demo
# Author: Sam Barba
# Created 11/09/2021

import random
import tkinter as tk

TEXT_LEN = 1472

text = ""

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def generate_text():
	global text, output_txt

	text = "".join([random.choice(list("ABCDE")) for _ in range(TEXT_LEN)])
	output_txt.configure(state="normal")
	output_txt.delete("1.0", tk.END)
	output_txt.insert("1.0", text)
	output_txt.tag_add("center", "1.0", tk.END)
	output_txt.configure(state="disabled")

	kmp()

def kmp(*args):
	global sv, text, output_txt, result_lbl

	pattern = sv.get().upper()
	len_t, len_p = len(text), len(pattern)

	# Colour output all black, before highlighting matched positions in blue
	output_txt.tag_remove("fill_black", "1.0", tk.END)
	output_txt.tag_add("fill_black", "1.0", tk.END)
	output_txt.tag_config("fill_black", foreground="black")

	# Clear all previous (if any) highlight tags
	for i in range(TEXT_LEN):
		output_txt.tag_remove(f"highlight{i}", "1.0", tk.END)

	if len_p == 0:
		result_lbl.configure(text="Pattern length must be > 0")
		return
	elif len_p > len_t:
		result_lbl.configure(text="Pattern must not be longer than text")
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
		output_txt.tag_add(f"highlight{pos}", f"1.{pos}", f"1.{pos + len(pattern)}")
		output_txt.tag_config(f"highlight{pos}", background="#20a020")

	result_lbl.configure(text=f"{len(positions)} results found:")

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

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
	root = tk.Tk()
	root.title("KMP demo")
	root.configure(width=650, height=750, bg="#141414")
	root.eval("tk::PlaceWindow . center")

	button = tk.Button(root, text="Generate random text", font="consolas", command=lambda: generate_text())
	button.place(relwidth=0.35, relheight=0.05, relx=0.5, rely=0.08, anchor="center")

	frame = tk.Frame(root, bg="#0080ff")
	frame.place(relwidth=0.5, relheight=0.14, relx=0.5, rely=0.21, anchor="center")

	enter_pattern_lbl = tk.Label(frame, text="Enter pattern to search:", font="consolas", bg="#0080ff")
	enter_pattern_lbl.place(relwidth=0.9, relheight=0.25, relx=0.5, rely=0.3, anchor="center")

	sv = tk.StringVar(value="ABC")
	sv.trace_add(mode="write", callback=kmp)

	pattern_entry = tk.Entry(frame, textvariable=sv,font="consolas", justify="center")
	pattern_entry.place(relwidth=0.5, relheight=0.26, relx=0.5, rely=0.62, anchor="center")

	result_lbl = tk.Label(root, font="consolas", bg="#141414", fg="#dcdcdc")
	result_lbl.place(relwidth=0.8, relheight=0.04, relx=0.5, rely=0.32, anchor="center")
	output_txt = tk.Text(root, bg="#dcdcdc", font="consolas", state="disabled")
	output_txt.tag_configure("center", justify="center")
	output_txt.place(relwidth=0.9, relheight=0.6, relx=0.5, rely=0.65, anchor="center")

	generate_text()

	root.mainloop()
