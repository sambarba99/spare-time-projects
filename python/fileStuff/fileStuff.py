# File Stuff
# Author: Sam Barba
# Created 09/01/2019

import os
import shutil

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
			print("File inside:", f)
			n += 1
			filePath = folderName + "\\" + str(f)
			try:
				pathSize += os.path.getsize(filePath)
			except:
				print("An exception occurred")
		print()

	print("--------------------------------------------------")
	print("\nPath:", path)
	print("Num. files in path:", n)

	suffixArr = ["bytes","KB","MB","GB"]
	pos = 0
	while pathSize > 1024 and pos < 3:
		pathSize /= 1024
		pos += 1

	print("Total size of path: {} {}".format(round(pathSize, 2), suffixArr[pos]))

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# os.getcwd() = current working directory
# os.makedirs("C:\\Users\\Sam Barba\\Desktop\\newFolder\\newFolder1") = make new folder inside new folder
# os.remove(filePath) = delete file in that path
# shutil.copy(sourcePath, destinationPath) = copies a file from source to destination
# shutil.copytree(sourcePath, backupPath) = creates 'backupPath' and copies sourcePath to this
# shutil.move(sourcePath, destinationPath) = moves sourcePath to destinationPath
# shutil.rmtree(path) = deletes all files at path

pathToExplore = "C:\\Users\\Sam Barba\\Desktop"

fileWalk(pathToExplore)

#N = 2**10000 #fact(991)
# writing a big number to a text file
#bigNumFile = open("C:\\Users\\Sam Barba\\Desktop\\bigNum.txt", "w")
#bigNumFile.write(str(N))
#bigNumFile.close()
