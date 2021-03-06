"""
File path walker

Author: Sam Barba
Created 09/01/2019
"""

import os
from time import perf_counter
import tkinter as tk

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def file_walk(path):
	"""Depth-first search of path, summing up file sizes"""

	global discovered_files_output_txt, path_size_output_lbl

	if not os.path.exists(path):
		discovered_files_output_txt.config(state='normal')
		discovered_files_output_txt.delete('1.0', tk.END)
		discovered_files_output_txt.config(state='disabled')
		path_size_output_lbl.config(text="That path doesn't exist!")
		return

	discovered_files = []
	n = path_size = 0

	start = perf_counter()

	if os.path.isfile(path):
		n, path_size = 1, os.path.getsize(path)

	for folder_name, subfolders, filenames in os.walk(path):
		discovered_files.append(f'Looking in: {folder_name}')
		for s in subfolders:
			discovered_files.append(f'Found subfolder: {s}')
		for f in filenames:
			file_path = folder_name + '\\' + str(f)
			file_size = os.path.getsize(file_path)
			path_size += file_size
			file_size, suffix = get_suffix(file_size)
			discovered_files.append(f'File inside: {f} ({file_size:.2f} {suffix})')
		n += len(filenames)

	interval = perf_counter() - start

	path_size, suffix = get_suffix(path_size)

	path_size_output = f'{n} files in path (size = {path_size:.2f} {suffix})\n' \
		f'Walked in {(1000 * interval):.0f} ms'

	discovered_files_output_txt.config(state='normal')
	discovered_files_output_txt.delete('1.0', tk.END)
	discovered_files_output_txt.insert('1.0', '\n'.join(discovered_files))
	discovered_files_output_txt.tag_add('center', '1.0', tk.END)
	discovered_files_output_txt.config(state='disabled')

	path_size_output_lbl.config(text=path_size_output)

def get_suffix(path_size):
	suffix_arr = ['bytes', 'KB', 'MB', 'GB']
	idx = 0

	while path_size >= 1024 and idx < 3:
		path_size /= 1024
		idx += 1

	return path_size, suffix_arr[idx]

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	root = tk.Tk()
	root.title('File Path Walker')
	root.config(width=600, height=680, bg='#000045')
	root.eval('tk::PlaceWindow . center')

	enter_path_lbl = tk.Label(root, text='Enter a file path:', font='consolas', bg='#000045', fg='white')
	enter_path_lbl.place(relwidth=0.9, relheight=0.04, relx=0.5, rely=0.06, anchor='center')

	entry_box = tk.Entry(root, font='consolas', justify='center')
	entry_box.place(relwidth=0.9, relheight=0.05, relx=0.5, rely=0.11, anchor='center')
	entry_box.insert(0, os.path.expanduser('~') + '\\Desktop')

	button = tk.Button(root, text='Walk', font='consolas', command=lambda: file_walk(entry_box.get()))
	button.place(relwidth=0.2, relheight=0.06, relx=0.5, rely=0.19, anchor='center')

	discovered_files_lbl = tk.Label(root, text='Discovered files:', font='consolas', bg='#000045', fg='white')
	discovered_files_lbl.place(relwidth=0.5, relheight=0.04, relx=0.5, rely=0.26, anchor='center')

	discovered_files_output_txt = tk.Text(root, bg='white', font='consolas', state='disabled')
	discovered_files_output_txt.tag_configure('center', justify='center')
	discovered_files_output_txt.place(relwidth=0.9, relheight=0.48, relx=0.5, rely=0.53, anchor='center')

	path_size_output_lbl = tk.Label(root, font='consolas', bg='white')
	path_size_output_lbl.place(relwidth=0.9, relheight=0.14, relx=0.5, rely=0.87, anchor='center')

	root.mainloop()
