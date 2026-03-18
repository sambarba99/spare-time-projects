"""
Utility for generating Conway's Game of Life patterns for C++ project

Use:
	1. Find a desired pattern from https://conwaylife.appspot.com/library/, https://conwaylife.com/wiki/, or https://conwaylife.com/patterns/
	2. Find either the ".rle" or the ".cells" link for the pattern
	3. Paste the URL into SOURCE_URL
	4. Run the code.

Author: Sam Barba
Created 28/11/2025
"""

from io import BytesIO
import re
import subprocess

import numpy as np
import pycurl


SOURCE_URL = 'https://conwaylife.com/patterns/hwss.rle'
FLIP_X = False
FLIP_Y = False
ROTATE_TIMES = 0
PATTERN_CONSTANT_NAME = 'HWSS'
PATTERN_DESCRIPTION = 'Heavyweight spaceship'


def decode_rle(rle):
	decoded = ''
	for count, char in re.findall(r'(\d+)?(\D)', rle):
		decoded	+= char * int(count) if count else char
	return decoded


def replace_every_20th_space(s):
	new_s = ''
	space_count = 0
	for c in s:
		if c == ' ':
			space_count += 1
			if space_count == 20:
				new_s += '\n\t'
				space_count = 0
			else:
				new_s += ' '
		else:
			new_s += c
	return new_s


if __name__ == '__main__':
	buffer = BytesIO()
	curl = pycurl.Curl()
	curl.setopt(curl.URL, SOURCE_URL)
	curl.setopt(curl.WRITEDATA, buffer)
	curl.perform()
	curl.close()
	html = buffer.getvalue().decode().strip()

	print(f'\nRaw HTML:\n\n{html}')

	if SOURCE_URL.endswith('.rle'):
		last_hash_idx = html.rfind('#')
		rle = html[last_hash_idx:].split('\n', 2)[-1]
		rle = re.sub(r'[\s!]', '', rle)
		source_str = decode_rle(rle)
		print(f'\nDecoded RLE:\n\n{source_str}')
	else:  # .cells
		last_exclamation_idx = html.rfind('!')
		source_str = html[last_exclamation_idx:].split('\n', 1)[-1]

	coords = []
	for y, line in enumerate(source_str.upper().replace('$', '\n').splitlines()):
		for x, char in enumerate(line):
			if char == 'O':
				coords.append((x, y))

	coords = np.array(coords)

	if FLIP_X:
		coords[:, 0] = coords[:, 0].max() - coords[:, 0]
	if FLIP_Y:
		coords[:, 1] = coords[:, 1].max() - coords[:, 1]
	if ROTATE_TIMES:
		for _ in range(ROTATE_TIMES):
			x = coords[:, 0]
			y = coords[:, 1]
			coords = np.stack((y.max() - y, x), axis=1)

	# Normalise grid coordinates so top-left is (0, 0)
	coords[:, 0] -= coords[:, 0].min()
	coords[:, 1] -= coords[:, 1].min()

	pattern_grid = [[' ' for _ in range(coords[:, 0].max() + 1)] for _ in range(coords[:, 1].max() + 1)]
	for x, y in coords:
		pattern_grid[y][x] = 'O'
	pattern_str = '\n'.join(''.join(row) for row in pattern_grid)
	print(f'\nPattern:\n\n{pattern_str}')

	coords = sorted(coords.tolist())
	coords_str = '{{' + '}, {'.join(str(c) for c in coords) + '}}'
	coords_str = replace_every_20th_space(re.sub(r'[\[\]]', '', coords_str))
	full_pattern_declaration = f'\nconst Pattern {PATTERN_CONSTANT_NAME} = {{"{PATTERN_DESCRIPTION}",\n\t{coords_str}}};\n'

	print(f'\nFull C++ pattern declaration:\n{full_pattern_declaration}')

	yn = input('\nCopy to clipboard (y/[n])? ')
	if yn.upper() == 'Y':
		p = subprocess.Popen(['clip'], stdin=subprocess.PIPE, text=True)
		p.communicate(full_pattern_declaration)
		print('Done')
