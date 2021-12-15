# File Stuff
# Author: Sam Barba
# Created 09/01/2019

import os
from time import perf_counter

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Depth-first search of path, summing up file sizes
def fileWalk(path):
	n = pathSize = 0

	if os.path.isfile(path):
		n, pathSize = 1, os.path.getsize(path)

	for folderName, subfolders, filenames in os.walk(path):
		print("Current folder:", folderName)
		for s in subfolders:
			print("Subfolder:", s)
		for f in filenames:
			n += 1
			filePath = folderName + "\\" + str(f)
			fileSize = os.path.getsize(filePath)
			pathSize += fileSize
			fileSize, suffix = getSuffix(fileSize)
			print(f"File inside: {f} ({fileSize} {suffix})")
		print()

	print("-" * 50)
	print(f"\n{n} files in {path}")

	pathSize, suffix = getSuffix(pathSize)
	print(f"Path size = {pathSize} {suffix}")

def getSuffix(pathSize):
	suffixArr = ["bytes", "KB", "MB", "GB"]
	idx = 0

	while pathSize >= 1024 and idx < 3:
		pathSize /= 1024
		idx += 1

	return round(pathSize, 2), suffixArr[idx]

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
fileWalk("C:\\Users\\Sam Barba\\Desktop\\Programs")
end = perf_counter()

print(f"Walked in {round(1000 * (end - start))} ms")
