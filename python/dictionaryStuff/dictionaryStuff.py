# Dictionary Stuff
# Author: Sam Barba
# Created 20/11/2018

WRITE_FILE = False

file = open("C:\\Users\\Sam Barba\\Desktop\\Programs\\programoutputs\\dictionary.txt", "r")
lines = file.readlines()
file.close()

lines = [word.replace("\n","") for word in lines]
numWords = len(lines)
numSpecificWords = 0
specificWordStr = ""

for word in lines:
	if "E" not in word:
		specificWordStr += ("\n" + word)
		numSpecificWords += 1

specificWordStr = specificWordStr[1:]

if WRITE_FILE:
	specificWordsFile = open("C:\\Users\\Sam Barba\\Desktop\\thing.txt", "w") # create new file with specific words
	specificWordsFile.write(specificWordStr)
	specificWordsFile.close()

print(specificWordStr)
print("\nNo. words:", numWords)
print("No. specific words:", numSpecificWords, "=", round(numSpecificWords / numWords * 100, 4), "%\n")
