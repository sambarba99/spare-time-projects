"""
File directory walker

Author: Sam Barba
Created 09/01/2019
"""

from datetime import datetime
from math import log
import multiprocessing
from pathlib import Path
from time import perf_counter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


UNITS = ['bytes', 'KB', 'MB', 'GB']
TBL_COLS = {
	'Path': {'width': 450, 'sort_order': ['Path']},
	'Accessed': {'width': 140, 'sort_order': ['Accessed', 'Path']},
	'Modified': {'width': 140, 'sort_order': ['Modified', 'Path']},
	'Created': {'width': 140, 'sort_order': ['Created', 'Path']},
	'Size': {'width': 80, 'sort_order': ['Size', 'Path']}
}

descending_orders = {col: False for col in TBL_COLS}
selected_dir = None


def select_dir():
	global selected_dir

	selected_dir = Path.home().as_posix()
	selected_dir = filedialog.askdirectory(
		title=f'{" " * 26}{"=" * 20}  Select a directory to walk  {"=" * 20}',
		initialdir=selected_dir
	)

	lbl_selected_dir.config(text=f'Selected directory: {selected_dir if selected_dir else "None"}')
	lbl_discovered_files.config(text='Discovered files')
	table.delete(*table.get_children())


def process_path(args):
	path, selected_dir = args
	fp = path.as_posix().removeprefix(selected_dir)
	stats = path.stat()
	accessed_str = datetime.fromtimestamp(stats.st_atime).strftime('%Y-%m-%d %H:%M:%S')
	modified_str = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
	created_str = datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
	file_size = stats.st_size

	return fp, accessed_str, modified_str, created_str, file_size


def get_file_stats(paths):
	def format_size_str(size_bytes):
		unit_idx = int(log(size_bytes, 1000)) if size_bytes > 0 else 0
		unit_idx = min(unit_idx, len(UNITS) - 1)
		size = size_bytes / (1000 ** unit_idx)
		unit = UNITS[unit_idx]
		return f'{int(size)} bytes' if unit == 'bytes' else f'{size:.2f} {unit}'


	row_values = []
	total_dir_size = 0

	if len(paths) < 5000:
		results = [process_path((path, f'{selected_dir}/')) for path in paths]
	else:
		args = [(path, f'{selected_dir}/') for path in paths]
		with multiprocessing.Pool() as pool:
			results = list(pool.imap_unordered(process_path, args, chunksize=64))

	for fp, accessed_str, modified_str, created_str, file_size in results:
		file_size_str = format_size_str(file_size)
		row_values.append((fp, accessed_str, modified_str, created_str, file_size_str))
		total_dir_size += file_size

	dir_size_str = format_size_str(total_dir_size)

	return row_values, dir_size_str


def print_tree(paths):
	def compress(name, node):
		parts = [name]
		while len(node) == 1:
			child_name, child_node = next(iter(node.items()))
			parts.append(child_name)
			node = child_node
		return '/'.join(parts), node

	def print_(node, prefix='', root=True):
		items = sorted(node.items(), key=lambda i: i[0].lower())  # Sort by name
		for idx, (name, child) in enumerate(items):
			name, child = compress(name, child)
			if root:
				print(name)
				extension = ''
			else:
				connector = '└── ' if idx == len(items) - 1 else '├── '
				print(prefix + connector + name)
				extension = '\t' if idx == len(items) - 1 else '│   '
			print_(child, prefix + extension, False)


	tree = dict()
	for path in paths:
		parts = path.as_posix().split('/')
		node = tree
		for part in parts:
			node = node.setdefault(part, dict())

	print_(tree)


def walk_dir():
	if not selected_dir:
		messagebox.showerror(title='Error', message='Select a directory')
		return

	start = perf_counter()

	paths = [p for p in Path(selected_dir).rglob('*') if p.is_file()]
	row_values, dir_size_str = get_file_stats(paths)

	interval = round(1000 * (perf_counter() - start))

	lbl_discovered_files.config(text=f'Discovered {len(row_values):,} files ({dir_size_str}) in {interval:,}ms')

	table.delete(*table.get_children())
	row_values.sort(key=lambda row: row[0].lower())  # Sort by path
	for vals in row_values:
		table.insert('', 'end', values=vals)

	table.yview_moveto(0)  # Scroll to top

	if len(paths) < 10_000:
		print('\nTree structure:\n')
		print_tree(paths)


def sort_column(col):
	sort_order = TBL_COLS[col]['sort_order']
	rows = []

	for row_id in table.get_children(''):
		if col == 'Size':
			size_str = table.set(row_id, 'Size')
			float_size_str, unit_str = size_str.split()
			float_size = float(float_size_str)
			unit_index = UNITS.index(unit_str)
			sort_key = unit_index, float_size
		else:
			sort_key = tuple(table.set(row_id, c).lower() for c in sort_order)

		rows.append((sort_key, row_id))

	rows.sort(key=lambda row: row[0], reverse=descending_orders[col])

	# Rearrange items into sorted positions
	for idx, (_, tkinter_row_id) in enumerate(rows):
		table.move(tkinter_row_id, '', idx)

	table.yview_moveto(0)

	# Reverse the sort order of this column for next time
	descending_orders[col] = not descending_orders[col]


if __name__ == '__main__':
	root = tk.Tk()
	root.title('File Directory Walker')
	root.config(width=1029, height=578, background='#101010')
	root.eval('tk::PlaceWindow . center')
	root.resizable(False, False)

	lbl_selected_dir = tk.Label(root, text='Selected directory: None',
		font='consolas 11', background='#101010', foreground='white'
	)
	lbl_selected_dir.place(width=1000, height=25, relx=0.5, y=27, anchor='center')

	button_select = tk.Button(root, text='Select directory', font='consolas 11', command=select_dir)
	button_select.place(width=170, height=35, x=468, y=63, anchor='center')
	button_walk = tk.Button(root, text='Walk', font='consolas 11', command=walk_dir)
	button_walk.place(width=90, height=35, x=602, y=63, anchor='center')

	lbl_discovered_files = tk.Label(root, text='Discovered files',
		font='consolas 11', background='#101010', foreground='white'
	)
	lbl_discovered_files.place(width=400, height=25, relx=0.5, y=108, anchor='center')

	table = ttk.Treeview(root, columns=list(TBL_COLS), show='headings')
	for col, col_dict in TBL_COLS.items():
		table.column(col, anchor='center', width=col_dict['width'])
		table.heading(col, text=col, command=lambda col_=col: sort_column(col_))
	ttk.Style().configure('Treeview.Heading', font=('Arial', 10, 'bold'))
	scrollbar = ttk.Scrollbar(root, command=table.yview)
	table.configure(yscrollcommand=scrollbar.set)
	table.place(height=426, x=506, y=340, anchor='center')
	scrollbar.place(height=426, x=990, y=340, anchor='center')

	root.mainloop()
