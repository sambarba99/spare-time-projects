"""
Program for solving 'Text Justification' LeetCode problem: https://leetcode.com/problems/text-justification/

Author: Sam Barba
Created 03/09/2022
"""

STRING = 'Python is a high-level, general-purpose programming language. Its design philosophy emphasizes' \
		 ' code readability with the use of significant indentation. Python is dynamically-typed and ' \
		 'garbage-collected. It supports multiple programming paradigms, including structured ' \
		 '(particularly procedural), object-oriented and functional programming. It is often described ' \
		 'as a "batteries included" language due to its comprehensive standard library. Guido van Rossum' \
		 ' began working on Python in the late 1980s as a successor to the ABC programming language and ' \
		 'first released it in 1991 as Python 0.9.0. Python 2.0 was released in 2000 and introduced new ' \
		 'features such as list comprehensions, cycle-detecting garbage collection, reference counting, ' \
		 'and Unicode support. Python 3.0, released in 2008, was a major revision that is not completely' \
		 ' backward-compatible with earlier versions. Python 2 was discontinued with version 2.7.18 in ' \
		 '2020. Python consistently ranks as one of the most popular programming languages.'

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def justify(words, max_width):
	n_words = len(words)
	start_idx = 0
	res = []

	while True:
		if (counter := start_idx) >= n_words:
			break
		line_len = 0

		# Find the start and end word indices for one line
		while counter < n_words:
			line_len += len(words[counter])
			if counter != start_idx:
				line_len += 1
			if line_len > max_width:
				break
			counter += 1

		# Justify one line
		if counter != n_words:
			end_idx = counter - 1
			if start_idx == end_idx:  # One word in line
				line = words[start_idx] + ' ' * (max_width - len(words[start_idx]))
			else:  # Many words in line
				line_len -= len(words[counter]) + 1
				word_num = end_idx - start_idx + 1
				extra_spaces = max_width - (line_len - (word_num - 1))
				basic_pad_spaces = extra_spaces // (word_num - 1)
				addition_pad_spaces = extra_spaces % (word_num - 1)
				line = ''
				for i in range(start_idx, counter - 1):
					line += words[i] + ' ' * basic_pad_spaces
					if i - start_idx < addition_pad_spaces:
						line += ' '
				line += words[counter - 1]
		else:  # Last line
			line = ' '.join(words[i] for i in range(start_idx, n_words))

		res.append(line)
		start_idx = counter

	return '\n'.join(res)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	print(justify('one two three four five six seven eight nine ten'.split(), 24))
	print()
	print(justify(STRING.split(), 150))
