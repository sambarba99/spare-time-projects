# KMP algorithm demo
# Author: Sam Barba
# Created 11/09/2021

import random

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def kmp(text, pattern):
	len_t, len_p = len(text), len(pattern)

	if len_p > len_t:
		raise ValueError("Pattern longer than text")

	# Create array for holding longest prefix suffix values for pattern
	lps = populate_lps(pattern)

	positions = []

	i = j = 0
	while i < len_t:
		if text[i] == pattern[j]:
			i += 1
			j += 1

		if j == len_p:
			positions.append(i - j)
			j = lps[j - 1]
		elif i < len_t and text[i] != pattern[j]:
			if j != 0:
				j = lps[j - 1]
			else:
				i += 1

	return positions if positions else "None"

def populate_lps(pattern):
	lps = [0] * len(pattern)
	i, length = 1, 0  # Length of previous longest prefix suffix

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

text = "".join([chr(random.randint(65, 67)) for _ in range(30)])
pattern = "".join([chr(random.randint(65, 67)) for _ in range(3)])
result = kmp(text, pattern)

print(f"   Text: {text}\nPattern: {pattern}\n\nPositions: {result}")
