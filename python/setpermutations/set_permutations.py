"""
Permutations demo

Author: Sam Barba
Created 08/09/2021
"""

import tkinter as tk

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def calculate(*_):
	char_set = sv.get()

	permutation_results = []
	permutations(list(char_set), 0, len(char_set) - 1, permutation_results)
	permutation_results = sorted(list(set(permutation_results)))

	permutation_repetition_results = []
	permutations_with_repetition_length_k(list(set(char_set)), len(char_set), permutation_repetition_results)
	permutation_repetition_results.sort()

	output_no_reps = f'{max(1, len(permutation_results))} unique permutations\n' \
		f"(no repetition) of '{char_set}':\n" \
		f"{', '.join(permutation_results)}"

	output_with_reps = f'{len(permutation_repetition_results)} permutations\n' \
		f"(with repetition) of '{char_set}':\n" \
		f"{', '.join(permutation_repetition_results)}"

	output_no_reps_txt.config(state='normal')
	output_no_reps_txt.delete('1.0', tk.END)
	output_no_reps_txt.insert('1.0', output_no_reps)
	output_no_reps_txt.tag_add('center', '1.0', tk.END)
	output_no_reps_txt.config(state='disabled')

	output_with_reps_txt.config(state='normal')
	output_with_reps_txt.delete('1.0', tk.END)
	output_with_reps_txt.insert('1.0', output_with_reps)
	output_with_reps_txt.tag_add('center', '1.0', tk.END)
	output_with_reps_txt.config(state='disabled')

def permutations(char_set, lo, hi, results):
	if lo == hi:
		results.append(''.join(char_set))

	for i in range(lo, hi + 1):
		char_set[lo], char_set[i] = char_set[i], char_set[lo]
		permutations(char_set, lo + 1, hi, results)
		char_set[lo], char_set[i] = char_set[i], char_set[lo]

def permutations_with_repetition_length_k(char_set, k, permutation_repetition_results, prefix=''):
	if k == 0:
		permutation_repetition_results.append(prefix)
	else:
		for c in char_set:
			new_prefix = prefix + c
			permutations_with_repetition_length_k(char_set, k - 1, permutation_repetition_results, new_prefix)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	root = tk.Tk()
	root.title('Permutations demo')
	root.config(width=700, height=600, bg='#000024')
	root.eval('tk::PlaceWindow . center')

	enter_word_lbl = tk.Label(root, text='Enter a word:', font='consolas', bg='#000024', fg='white')

	sv = tk.StringVar(value='hello')
	sv.trace_add(mode='write', callback=calculate)

	entry_box = tk.Entry(root, textvariable=sv, font='consolas', justify='center')

	result_no_reps_lbl = tk.Label(root, text='No repetition:', font='consolas', bg='#000024', fg='white')
	output_no_reps_txt = tk.Text(root, font='consolas', bg='white', state='disabled')
	output_no_reps_txt.tag_configure('center', justify='center')

	result_with_reps_lbl = tk.Label(root, text='With repetition:', font='consolas', bg='#000024', fg='white')
	output_with_reps_txt = tk.Text(root, bg='white', font='consolas', state='disabled')
	output_with_reps_txt.tag_configure('center', justify='center')

	enter_word_lbl.place(relwidth=0.3, relheight=0.05, relx=0.5, rely=0.06, anchor='center')
	entry_box.place(relwidth=0.3, relheight=0.05, relx=0.5, rely=0.12, anchor='center')
	result_no_reps_lbl.place(relwidth=0.9, relheight=0.04, relx=0.5, rely=0.19, anchor='center')
	output_no_reps_txt.place(relwidth=0.9, relheight=0.33, relx=0.5, rely=0.38, anchor='center')
	result_with_reps_lbl.place(relwidth=0.9, relheight=0.04, relx=0.5, rely=0.58, anchor='center')
	output_with_reps_txt.place(relwidth=0.9, relheight=0.33, relx=0.5, rely=0.77, anchor='center')

	calculate()

	root.mainloop()
