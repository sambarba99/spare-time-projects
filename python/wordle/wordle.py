"""
Wordle game with AI assistance (solves with ~3.5 attempts on average)

Author: Sam Barba
Created 21/03/2022

Controls:
Enter: submit attempt
Backspace: delete last char
Tab: AI assistance
Space: reset when game over
"""

import numpy as np
import pygame as pg
import sys
from time import sleep

WORD_LEN = 5
MAX_ATTEMPTS = 6

CELL_SIZE = 70
GRID_OFFSET = 50
GAP = 5
BACKGROUND = (12, 12, 12)
LIGHT_GREY = (130, 130, 130)
GREY = (50, 50, 50)     # Letter not in word
YELLOW = (255, 160, 0)  # Letter in word, but wrong position
GREEN = (60, 160, 60)   # Letter in word and in correct position

KEYBOARD_ROWS = ['QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM']
KEY_SIZE = 34
KEY_GAP = 2

attempts = [[''] * WORD_LEN for _ in range(MAX_ATTEMPTS)]
attempt_num = col_num = 0
green_letters, yellow_letters, grey_letters = [], [], []
game_over = False
target_word = scene = None

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def generate_pattern_dict(word_list):
	"""
	E.g. If word_list = ['BEARS', 'CRANE', 'WEARY']:

	pattern_dict = generate_pattern_dict(word_list)

	pattern_dict['BEARS'][(2, 2, 2, 2, 2)] = 'BEARS'

	pattern_dict['CRANE'][(0, 1, 2, 0, 1)] = {'BEARS', 'WEARY'}
	"""

	pattern_dict = {w: dict() for w in word_list}

	for w1 in word_list:
		for w2 in word_list:
			pattern = get_information(w1, w2, update_keyboard=False)
			if pattern not in pattern_dict[w1]:
				pattern_dict[w1][pattern] = set()
			pattern_dict[w1][pattern].add(w2)

	return pattern_dict

def calculate_word_entropies(words_left, pattern_dict, all_patterns):
	def calculate_entropy(y):
		if len(y) <= 1: return 0

		counts = np.bincount(y)
		probs = counts[np.nonzero(counts)] / len(y)  # np.nonzero ensures that we're not doing log(0) after

		return -(probs * np.log2(probs)).sum()

	entropies = {}

	for word in words_left:
		counts = []
		for pattern in all_patterns:
			if pattern in pattern_dict[word]:
				matches = pattern_dict[word][pattern]
				matches = matches.intersection(words_left)
				counts.append(len(matches))
			else:
				counts.append(0)

		entropies[word] = calculate_entropy(counts)

	return entropies

def generate_all_info_patterns(permutations, k=WORD_LEN, prefix=''):
	if k == 0:
		permutations.append(tuple([int(c) for c in prefix]))
	else:
		for c in '012':  # 0/1/2 = grey/yellow/green
			new_prefix = prefix + c
			generate_all_info_patterns(permutations, k - 1, new_prefix)

def get_information(user_word, target_word, update_keyboard):
	"""
	Get colour information given user_word and target_word

	E.g. user_word = CRANE and target_word = BEARS: info = (0, 1, 2, 0, 1)
	"""

	if len(user_word) != len(target_word): return (0,) * WORD_LEN

	# 0 = grey, 1 = yellow, 2 = green
	col_codes = [0] * WORD_LEN
	temp_user_word = list(user_word)
	temp_target_word = list(target_word)

	# Check for green letters
	for i in range(WORD_LEN):
		if user_word[i] == target_word[i]:
			# Mark position as checked (-) so it's not triggered by yellow pass
			temp_user_word[i] = temp_target_word[i] = '-'
			col_codes[i] = 2
			if update_keyboard:
				green_letters.append(user_word[i])

	# Check for yellow letters
	for i in range(WORD_LEN):
		if temp_user_word[i] == '-':
			continue
		elif temp_user_word[i] in temp_target_word:
			# Mark letter as checked in case it's repeated again
			temp_target_word[temp_target_word.index(temp_user_word[i])] = '-'
			temp_user_word[i] = '-'
			col_codes[i] = 1
			if update_keyboard:
				yellow_letters.append(user_word[i])
		elif update_keyboard:
			grey_letters.append(user_word[i])

	return tuple(col_codes)

def draw_grid(colour_current_row, status=None):
	scene.fill(BACKGROUND)
	font = pg.font.SysFont('arial bold', 40)

	# Draw grid of attempts
	for y in range(MAX_ATTEMPTS):
		user_word = ''.join(attempts[y])
		if y < attempt_num or colour_current_row:
			row_colours = get_information(user_word, target_word, update_keyboard=True)
		else:
			row_colours = [0] * WORD_LEN

		for x in range(WORD_LEN):
			if y < attempt_num or (len(user_word) == WORD_LEN and colour_current_row):
				match row_colours[x]:
					case 2: fill_col = GREEN
					case 1: fill_col = YELLOW
					case _: fill_col = GREY

				pg.draw.rect(scene, fill_col, pg.Rect(x * (CELL_SIZE + GAP) + GRID_OFFSET,
					y * (CELL_SIZE + GAP) + GRID_OFFSET, CELL_SIZE, CELL_SIZE))
			else:
				if y == attempt_num and x < col_num:
					pg.draw.rect(scene, LIGHT_GREY,
						pg.Rect(x * (CELL_SIZE + GAP) + GRID_OFFSET,
							y * (CELL_SIZE + GAP) + GRID_OFFSET, CELL_SIZE, CELL_SIZE),
						width=2)
				else:
					pg.draw.rect(scene, GREY,
						pg.Rect(x * (CELL_SIZE + GAP) + GRID_OFFSET,
							y * (CELL_SIZE + GAP) + GRID_OFFSET, CELL_SIZE, CELL_SIZE),
						width=2)

			cell_lbl = font.render(attempts[y][x], True, (255, 255, 255))
			lbl_rect = cell_lbl.get_rect(center=((x + 0.5) * CELL_SIZE + x * GAP + GRID_OFFSET,
				(y + 0.5) * CELL_SIZE + y * GAP + GRID_OFFSET))
			scene.blit(cell_lbl, lbl_rect)

	# Draw keyboard
	key_font = pg.font.SysFont('arial', 20)

	for y, row in enumerate(KEYBOARD_ROWS):
		match y:
			case 0: offset = 56
			case 1: offset = 72
			case _: offset = 108

		for x, letter in enumerate(row):
			match letter:
				case letter if letter in green_letters: fill_col = GREEN
				case letter if letter in yellow_letters: fill_col = YELLOW
				case letter if letter in grey_letters: fill_col = GREY
				case _: fill_col = LIGHT_GREY

			pg.draw.rect(scene, fill_col, pg.Rect(x * (KEY_SIZE + KEY_GAP) + offset,
				530 + y * (KEY_SIZE + KEY_GAP), KEY_SIZE, KEY_SIZE))
			key_lbl = key_font.render(letter, True, (255, 255, 255))
			lbl_rect = key_lbl.get_rect(center=((x + 0.5) * KEY_SIZE + x * KEY_GAP + offset,
				530 + (y + 0.5) * KEY_SIZE + y * KEY_GAP))
			scene.blit(key_lbl, lbl_rect)

	# Draw status label
	status_font = pg.font.SysFont('consolas', 18)

	if status is None:
		status_lbl = status_font.render(f'Attempt {attempt_num + 1}/{MAX_ATTEMPTS}', True, (220, 220, 220))
	else:
		status_lbl = status_font.render(status, True, (255, 255, 255))

	x = WORD_LEN * (CELL_SIZE + GAP) + 2 * GRID_OFFSET
	lbl_rect = status_lbl.get_rect(center=(x / 2, 675))
	scene.blit(status_lbl, lbl_rect)

	pg.display.update()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	global attempts, attempt_num, col_num, target_word, game_over, scene, \
		green_letters, yellow_letters, grey_letters

	pg.init()
	pg.display.set_caption('Wordle')
	scene = pg.display.set_mode((WORD_LEN * (CELL_SIZE + GAP) + 2 * GRID_OFFSET,
		MAX_ATTEMPTS * (CELL_SIZE + GAP) + 2 * GRID_OFFSET + 180))

	draw_grid(colour_current_row=False, status='Generating pattern dictionary...')

	all_words = np.genfromtxt('five_letter_words.txt', dtype=str, delimiter='\n')
	pattern_dict = generate_pattern_dict(all_words)

	draw_grid(colour_current_row=False)

	# Generate all possible patterns (3^5 = 243) of green/yellow/grey info
	# i.e. (0, 0, 0, 0, 0) ... (2, 2, 2, 2, 2)
	all_patterns = []
	generate_all_info_patterns(all_patterns)

	words_left = set(all_words)

	# If using AI to solve online wordle...

	choice = input('Solving online wordle? (Y/N)\n>>> ').upper()
	if choice == 'Y':
		# Suggest start word with highest entropy
		entropies = calculate_word_entropies(words_left, pattern_dict, all_patterns)
		attempt_word = max(entropies, key=entropies.get)

		i = 1
		while i <= MAX_ATTEMPTS:
			print(f"\nAttempt {i}/{MAX_ATTEMPTS}: try '{attempt_word}'")
			info = input(f"Enter the info (0/1/2 for grey/yellow/green) for '{attempt_word}' (or W if won)\n>>> ").upper()

			if info == 'W': break

			info = tuple([int(c) for c in info])

			if info not in pattern_dict[attempt_word]:
				print('\nNo words left in dictionary to try :(')
				break

			matches = pattern_dict[attempt_word][info]
			words_left = words_left.intersection(matches)

			if len(words_left) == 0:
				print('\nNo words left in dictionary to try :(')
				break

			# Suggest word left with highest entropy
			entropies = calculate_word_entropies(words_left, pattern_dict, all_patterns)
			attempt_word = max(entropies, key=entropies.get)

			i += 1

	# If playing game here...
	target_word = np.random.choice(all_words)
	words_left = set(all_words)

	while True:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				sys.exit()

			elif event.type == pg.KEYDOWN:
				if game_over and event.key != pg.K_SPACE: continue

				if game_over and event.key == pg.K_SPACE:  # Reset if game over
					attempts = [[''] * WORD_LEN for _ in range(MAX_ATTEMPTS)]
					attempt_num = col_num = 0
					green_letters, yellow_letters, grey_letters = [], [], []
					game_over = False
					words_left = set(all_words)
					target_word = np.random.choice(all_words)
					draw_grid(colour_current_row=False)

				elif 97 <= event.key <= 122:  # a - z
					if attempt_num < MAX_ATTEMPTS and col_num < WORD_LEN:
						attempts[attempt_num][col_num] = chr(event.key - 32)  # -32 to capitalise
						col_num += 1
						draw_grid(colour_current_row=False)

				elif event.key == pg.K_RETURN:  # Submit attempt
					if col_num != WORD_LEN: continue

					user_word = ''.join(attempts[attempt_num])

					if user_word not in all_words:
						draw_grid(colour_current_row=False, status=f"'{user_word}' not in word list!")
						sleep(1.5)
						attempts[attempt_num] = [''] * WORD_LEN
						col_num = 0
						draw_grid(colour_current_row=False)
					elif user_word == target_word:
						draw_grid(colour_current_row=True, status=f'You win! SPACE to reset.')
						game_over = True
					elif user_word != target_word and attempt_num == MAX_ATTEMPTS - 1:
						draw_grid(colour_current_row=True, status=f"You lose! The target was '{target_word}'.")
						game_over = True
					else:
						attempt_num += 1
						col_num = 0
						draw_grid(colour_current_row=True)

						# Filter list of remaining possible words
						info = get_information(user_word, target_word, update_keyboard=False)
						words = pattern_dict[user_word][info]
						words_left = words_left.intersection(words)

				elif event.key == pg.K_BACKSPACE:  # Delete last char
					col_num = max(col_num - 1, 0)
					attempts[attempt_num][col_num] = ''
					draw_grid(colour_current_row=False)

				elif event.key == pg.K_TAB:  # AI assistance
					# Suggest word left with highest entropy
					entropies = calculate_word_entropies(words_left, pattern_dict, all_patterns)
					attempt_word = max(entropies, key=entropies.get)

					draw_grid(colour_current_row=False, status=f"Try '{attempt_word}'...")
					sleep(1)
					draw_grid(colour_current_row=False)

if __name__ == '__main__':
	main()
