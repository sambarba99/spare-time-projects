# KMP algorithm demo
# Author: Sam Barba
# Created 11/09/2021

import random

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def kmp(text, pattern):
	lenT, lenP = len(text), len(pattern)

	if lenP > lenT:
		raise ValueError("Pattern longer than text")

	# create array for holding longest prefix suffix values for pattern
	lps = populateLPS(pattern)

	positions = []

	i = j = 0
	while i < lenT:
		if text[i] == pattern[j]:
			i += 1
			j += 1

		if j == lenP:
			positions.append(i - j)
			j = lps[j - 1]
		elif i < lenT and text[i] != pattern[j]:
			if j != 0:
				j = lps[j - 1]
			else:
				i += 1

	return positions if positions else "None"

def populateLPS(pattern):
	lps = [0] * len(pattern)
	i, length = 1, 0 # length of previous longest prefix suffix

	while i < len(pattern):
		if pattern[i] == pattern[length]:
			length += 1
			lps[i] = length
			i += 1
		else:
			if length != 0:
				length = lps[length - 1]
			else:
				lps[i] = 0
				i += 1

	return lps

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

while True:
	text = "".join([chr(random.randint(65, 67)) for i in range(20)])
	pattern = "".join([chr(random.randint(65, 67)) for i in range(3)])

	print("   Text: {}\nPattern: {}\n\nPositions: {}".format(text, pattern, kmp(text, pattern)))

	choice = input("\nEnter to continue or X to exit: ").upper()
	if len(choice) > 0 and choice[0] == 'X':
		break
	print()
