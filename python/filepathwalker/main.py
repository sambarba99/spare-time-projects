"""
File path walker

Author: Sam Barba
Created 09/01/2019
"""

import os
from time import perf_counter
import tkinter as tk


def file_walk(path):
	"""Depth-first search of path, summing up file sizes"""

	def get_suffix(path_size):
		suffix_arr = ['bytes', 'KB', 'MB', 'GB']
		idx = 0

		while path_size >= 1024 and idx < 3:
			path_size /= 1024
			idx += 1

		return path_size, suffix_arr[idx]


	if not os.path.exists(path):
		discovered_files_output_txt.config(state='normal')
		discovered_files_output_txt.delete('1.0', tk.END)
		discovered_files_output_txt.config(state='disabled')
		path_size_output_lbl.config(text="That path doesn't exist!")
		return

	discovered_files = []
	n_files = path_size = 0

	start = perf_counter()

	if os.path.isfile(path):
		n_files, path_size = 1, os.path.getsize(path)

	for folder_name, subfolders, filenames in os.walk(path):
		discovered_files.append(f'Looking in: {folder_name}')
		for subf in subfolders:
			discovered_files.append(f'Found subfolder: {subf}')
		for fname in filenames:
			file_path = fr'{folder_name}\{fname}'
			file_size = os.path.getsize(file_path)
			path_size += file_size
			file_size, suffix = get_suffix(file_size)
			discovered_files.append(f'Found file: {fname} ({file_size:.2f} {suffix})')
		discovered_files.append('')
		n_files += len(filenames)

	end = perf_counter()
	interval = round(1000 * (end - start))

	path_size, suffix = get_suffix(path_size)

	path_size_output = f'{n_files:,} files in path ({path_size:.2f} {suffix})' \
		f'\nWalked in {interval} ms'

	discovered_files_output_txt.config(state='normal')
	discovered_files_output_txt.delete('1.0', tk.END)
	discovered_files_output_txt.insert('1.0', '\n'.join(discovered_files))
	discovered_files_output_txt.tag_add('center', '1.0', tk.END)
	discovered_files_output_txt.config(state='disabled')

	path_size_output_lbl.config(text=path_size_output)


if __name__ == '__main__':
	root = tk.Tk()
	root.title('File Path Walker')
	root.config(width=600, height=680, bg='#000024')
	root.eval('tk::PlaceWindow . center')
	root.resizable(False, False)

	enter_path_lbl = tk.Label(root, text='Enter a file path:', font='consolas', bg='#000024', fg='white')

	entry_box = tk.Entry(root, font='consolas', justify='center')
	entry_box.insert(0, os.path.expanduser('~') + r'\Desktop')

	button = tk.Button(root, text='Walk', font='consolas', command=lambda: file_walk(entry_box.get()))

	discovered_files_lbl = tk.Label(root, text='Discovered files:', font='consolas', bg='#000024', fg='white')
	discovered_files_output_txt = tk.Text(root, bg='white', font='consolas', state='disabled')
	discovered_files_output_txt.tag_configure('center', justify='center')
	path_size_output_lbl = tk.Label(root, font='consolas', bg='white')

	enter_path_lbl.place(width=200, height=25, relx=0.5, y=41, anchor='center')
	entry_box.place(width=540, height=32, relx=0.5, y=75, anchor='center')
	button.place(width=120, height=40, relx=0.5, y=129, anchor='center')
	discovered_files_lbl.place(width=200, height=25, relx=0.5, y=177, anchor='center')
	discovered_files_output_txt.place(width=540, height=330, relx=0.5, y=360, anchor='center')
	path_size_output_lbl.place(width=540, height=100, relx=0.5, y=592, anchor='center')

	root.mainloop()
