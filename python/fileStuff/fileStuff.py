# File Stuff
# Author: Sam Barba
# Created 09/01/2019

import os
from time import perf_counter

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Depth-first search of path, summing up file sizes
def file_walk(path):
	n = path_size = 0

	if os.path.isfile(path):
		n, path_size = 1, os.path.getsize(path)

	for folder_name, subfolders, filenames in os.walk(path):
		print("Current folder:", folder_name)
		for s in subfolders:
			print("Subfolder:", s)
		for f in filenames:
			n += 1
			file_path = folder_name + "\\" + str(f)
			file_size = os.path.getsize(file_path)
			path_size += file_size
			file_size, suffix = get_suffix(file_size)
			print(f"File inside: {f} ({file_size} {suffix})")
		print()

	print("-" * 50)
	print(f"\n{n} files in {path}")

	path_size, suffix = get_suffix(path_size)
	print(f"Path size = {path_size} {suffix}")

def get_suffix(path_size):
	suffix_arr = ["bytes", "KB", "MB", "GB"]
	idx = 0

	while path_size >= 1024 and idx < 3:
		path_size /= 1024
		idx += 1

	return round(path_size, 2), suffix_arr[idx]

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# os.getcwd() = current working directory
# os.makedirs("C:\\Users\\Sam Barba\\Desktop\\newFolder\\newFolder1") = make newFolder1 inside newFolder
# shutil.copy(sourcePath, destinationPath) = copies a file from source to destination
# shutil.copytree(sourcePath, backupPath) = creates 'backupPath' and copies sourcePath to this
# shutil.move(sourcePath, destinationPath) = moves sourcePath to destinationPath
# shutil.rmtree(path) = deletes all files at path

start = perf_counter()
file_walk("C:\\Users\\Sam Barba\\Desktop\\Programs")
end = perf_counter()

print(f"Walked in {round(1000 * (end - start))} ms")
