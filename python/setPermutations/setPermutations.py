# Permutations demo
# Author: Sam Barba
# Created 08/09/2021

import tkinter as tk

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def calculate(*args):
	global sv, output_no_reps_txt, output_with_reps_txt

	char_set = sv.get()

	permutation_results = []
	permutations(len(char_set), list(char_set), permutation_results)
	permutation_results = sorted(list(set(permutation_results)))

	permutation_repetition_results = []
	permutations_with_repetition_length_k(list(set(char_set)), len(char_set), permutation_repetition_results)
	permutation_repetition_results.sort()

	output_no_reps = f"{len(permutation_results)} unique permutations\n" \
		f"(no repetition) of '{char_set}':\n" \
		f"{', '.join(permutation_results)}"

	output_with_reps = f"{len(permutation_repetition_results)} permutations\n" \
		f"(with repetition) of '{char_set}':\n" \
		f"{', '.join(permutation_repetition_results)}"

	output_no_reps_txt.configure(state="normal")
	output_no_reps_txt.delete("1.0", tk.END)
	output_no_reps_txt.insert("1.0", output_no_reps)
	output_no_reps_txt.tag_add("center", "1.0", tk.END)
	output_no_reps_txt.configure(state="disabled")

	output_with_reps_txt.configure(state="normal")
	output_with_reps_txt.delete("1.0", tk.END)
	output_with_reps_txt.insert("1.0", output_with_reps)
	output_with_reps_txt.tag_add("center", "1.0", tk.END)
	output_with_reps_txt.configure(state="disabled")

# Heap's algorithm for generating all permutations of n objects
def permutations(n, char_set, results):
	if n == 1:
		results.append("".join(char_set))
	else:
		for i in range(n):
			permutations(n - 1, char_set, results)
			if n % 2 == 0:
				char_set[i], char_set[n - 1] = char_set[n - 1], char_set[i]
			else:
				char_set[0], char_set[n - 1] = char_set[n - 1], char_set[0]

def permutations_with_repetition_length_k(char_set, k, permutation_repetition_results, prefix=""):
	if k == 0:
		permutation_repetition_results.append(prefix)
	else:
		for c in char_set:
			new_prefix = prefix + c
			permutations_with_repetition_length_k(char_set, k - 1, permutation_repetition_results, new_prefix)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

root = tk.Tk()
root.title("Permutations demo")
root.configure(width=700, height=600, bg="#141414")
root.eval("tk::PlaceWindow . center")

frame = tk.Frame(root, bg="#0080ff")
frame.place(relwidth=0.5, relheight=0.12, relx=0.5, rely=0.1, anchor="center")

enter_word_lbl = tk.Label(frame, text="Enter a word:", font="consolas", bg="#0080ff")
enter_word_lbl.place(relwidth=0.8, relheight=0.33, relx=0.5, rely=0.25, anchor="center")

sv = tk.StringVar(value="hello")
sv.trace_add(mode="write", callback=calculate)

entry_box = tk.Entry(frame, textvariable=sv, font="consolas", justify="center")
entry_box.place(relwidth=0.5, relheight=0.37, relx=0.5, rely=0.62, anchor="center")

result_no_reps_lbl = tk.Label(root, text="No repetition:", font="consolas", bg="#141414", fg="#dcdcdc")
result_no_reps_lbl.place(relwidth=0.9, relheight=0.04, relx=0.5, rely=0.21, anchor="center")

output_no_reps_txt = tk.Text(root, font="consolas", bg="#dcdcdc", state="disabled")
output_no_reps_txt.tag_configure("center", justify="center")
output_no_reps_txt.place(relwidth=0.9, relheight=0.33, relx=0.5, rely=0.4, anchor="center")

result_with_reps_lbl = tk.Label(root, text="With repetition:", font="consolas", bg="#141414", fg="#dcdcdc")
result_with_reps_lbl.place(relwidth=0.9, relheight=0.04, relx=0.5, rely=0.6, anchor="center")

output_with_reps_txt = tk.Text(root, bg="#dcdcdc", font="consolas", state="disabled")
output_with_reps_txt.tag_configure("center", justify="center")
output_with_reps_txt.place(relwidth=0.9, relheight=0.33, relx=0.5, rely=0.79, anchor="center")

calculate()

root.mainloop()
