"""
File path walker

Author: Sam Barba
Created 09/01/2019
"""

from math import log
import os
from time import perf_counter
import tkinter as tk
from tkinter import filedialog, messagebox


path = None


def select_path():
	global path

	path = f"{os.path.expanduser('~')}/Desktop".replace('\\', '/')
	path = filedialog.askdirectory(
		title=f'{" " * 26}{"=" * 20}  Select a file path to walk  {"=" * 20}',
		initialdir=path
	)

	lbl_selected_path.config(text=f'Selected path: {path if path else "None"}')


def file_walk():
	"""Depth-first search of path, summing up file sizes"""

	def get_suffix(path_size_bytes):
		units = ['bytes', 'KB', 'MB', 'GB']
		unit_idx = int(log(path_size_bytes, 1000)) if path_size_bytes > 0 else 0
		unit_idx = min(unit_idx, len(units) - 1)

		return path_size_bytes / (1000 ** unit_idx), units[unit_idx]


	if not path:
		messagebox.showwarning(title='Error', message='Select a path')
		return

	discovered_files = []
	num_files = path_size = 0

	start = perf_counter()

	if os.path.isfile(path):
		num_files, path_size = 1, os.path.getsize(path)

	for folder_name, subfolders, filenames in os.walk(path):
		folder_name = folder_name.replace('\\', '/')
		discovered_files.append(f'Looking in: {folder_name}')
		for subf in subfolders:
			discovered_files.append(f'Found subfolder: {subf}')
		for fname in filenames:
			file_path = f'{folder_name}/{fname}'
			file_size = os.path.getsize(file_path)
			path_size += file_size
			file_size, suffix = get_suffix(file_size)
			discovered_files.append(f'Found file: {fname} ({file_size:.2f} {suffix})')
		discovered_files.append('')
		num_files += len(filenames)

	end = perf_counter()
	interval = round(1000 * (end - start))

	path_size, suffix = get_suffix(path_size)

	discovered_files_output_txt.config(state='normal')
	discovered_files_output_txt.delete('1.0', 'end')
	discovered_files_output_txt.insert('1.0', '\n'.join(discovered_files))
	discovered_files_output_txt.tag_add('center', '1.0', 'end')
	discovered_files_output_txt.config(state='disabled')

	path_size_output_lbl.config(
		text=f'Found {num_files:,} files ({path_size:.2f} {suffix})'
		f'\nWalked in {interval} ms'
	)


if __name__ == '__main__':
	root = tk.Tk()
	root.title('File Path Walker')
	root.config(width=600, height=660, background='#101010')
	root.eval('tk::PlaceWindow . center')
	root.resizable(False, False)

	lbl_selected_path = tk.Label(root, text='Selected path: None', font='consolas',
		background='#101010', foreground='white'
	)
	button_select_path = tk.Button(root, text='Select path', font='consolas', command=select_path)
	button_walk = tk.Button(root, text='Walk', font='consolas', command=file_walk)

	discovered_files_lbl = tk.Label(root, text='Discovered files:', font='consolas',
		background='#101010', foreground='white'
	)
	discovered_files_output_txt = tk.Text(root, background='white', font='consolas', state='disabled')
	discovered_files_output_txt.tag_configure('center', justify='center')
	path_size_output_lbl = tk.Label(root, font='consolas', background='white')

	lbl_selected_path.place(width=550, height=25, relx=0.5, y=30, anchor='center')
	button_select_path.place(width=130, height=35, x=232, y=75, anchor='center')
	button_walk.place(width=130, height=35, x=368, y=75, anchor='center')
	discovered_files_lbl.place(width=200, height=25, relx=0.5, y=120, anchor='center')
	discovered_files_output_txt.place(width=550, height=383, relx=0.5, y=330, anchor='center')
	path_size_output_lbl.place(width=550, height=95, relx=0.5, y=582, anchor='center')

	root.mainloop()
