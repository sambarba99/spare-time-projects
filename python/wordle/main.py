"""
Wordle game with information theory assistance (solves with ~3.5 attempts on average)

Author: Sam Barba
Created 21/03/2022

Controls:
Enter: submit attempt
Backspace: delete last char
Tab: use assistance
Space: reset when game over
"""

import sys

import numpy as np
import pygame as pg
from tqdm import tqdm


WORD_LEN = 5
MAX_ATTEMPTS = 6

CELL_SIZE = 70
GRID_OFFSET = 50
GAP = 5
BACKGROUND = (18, 18, 18)
GREY = (59, 59, 59)      # Letter not in word
YELLOW = (181, 159, 59)  # Letter in word, but wrong position
GREEN = (83, 141, 78)    # Letter in word and in correct position
LIGHT_GREY = (130, 130, 130)

KEYBOARD_ROWS = ['QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM']
KEY_SIZE = 34
KEY_GAP = 2

attempts = [[''] * WORD_LEN for _ in range(MAX_ATTEMPTS)]
attempt_num = col_num = 0
green_letters, yellow_letters, grey_letters = [], [], []
game_over = False
target_word = scene = None


def generate_pattern_dict(word_list):
	"""
	E.g. If word_list = ['BEARS', 'CRANE', 'WEARY']:

	pattern_dict = generate_pattern_dict(word_list)

	pattern_dict['BEARS'][(2, 2, 2, 2, 2)] = 'BEARS'

	pattern_dict['CRANE'][(0, 1, 2, 0, 1)] = {'BEARS', 'WEARY'}
	"""

	pattern_dict = {w: dict() for w in word_list}

	for w1 in tqdm(word_list, desc='Generating pattern dictionary', ascii=True):
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
		probs = counts[counts.nonzero()] / len(y)  # .nonzero ensures that we're not doing log(0) after

		return -(probs * np.log2(probs)).sum()


	entropies = dict()

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
	for i in range(MAX_ATTEMPTS):
		user_word = ''.join(attempts[i])
		if i < attempt_num or colour_current_row:
			row_colours = get_information(user_word, target_word, update_keyboard=True)
		else:
			row_colours = [0] * WORD_LEN

		for j in range(WORD_LEN):
			if i < attempt_num or (len(user_word) == WORD_LEN and colour_current_row):
				match row_colours[j]:
					case 2: fill_col = GREEN
					case 1: fill_col = YELLOW
					case _: fill_col = GREY

				pg.draw.rect(scene, fill_col, pg.Rect(j * (CELL_SIZE + GAP) + GRID_OFFSET,
					i * (CELL_SIZE + GAP) + GRID_OFFSET, CELL_SIZE, CELL_SIZE))
			else:
				if i == attempt_num and j < col_num:
					pg.draw.rect(scene, GREY,
						pg.Rect(j * (CELL_SIZE + GAP) + GRID_OFFSET,
							i * (CELL_SIZE + GAP) + GRID_OFFSET, CELL_SIZE, CELL_SIZE),
						width=2)
				else:
					pg.draw.rect(scene, GREY,
						pg.Rect(j * (CELL_SIZE + GAP) + GRID_OFFSET,
							i * (CELL_SIZE + GAP) + GRID_OFFSET, CELL_SIZE, CELL_SIZE),
						width=2)

			cell_lbl = font.render(attempts[i][j], True, (255, 255, 255))
			lbl_rect = cell_lbl.get_rect(center=((j + 0.5) * CELL_SIZE + j * GAP + GRID_OFFSET,
				(i + 0.5) * CELL_SIZE + i * GAP + GRID_OFFSET))
			scene.blit(cell_lbl, lbl_rect)

	# Draw keyboard
	key_font = pg.font.SysFont('arial', 20)

	for idx, row in enumerate(KEYBOARD_ROWS):
		match idx:
			case 0: offset = 56
			case 1: offset = 72
			case _: offset = 108

		for j, letter in enumerate(row):
			match letter:
				case letter if letter in green_letters: fill_col = GREEN
				case letter if letter in yellow_letters: fill_col = YELLOW
				case letter if letter in grey_letters: fill_col = GREY
				case _: fill_col = LIGHT_GREY

			pg.draw.rect(scene, fill_col, pg.Rect(j * (KEY_SIZE + KEY_GAP) + offset,
				530 + idx * (KEY_SIZE + KEY_GAP), KEY_SIZE, KEY_SIZE))
			key_lbl = key_font.render(letter, True, (255, 255, 255))
			lbl_rect = key_lbl.get_rect(center=((j + 0.5) * KEY_SIZE + j * KEY_GAP + offset,
				530 + (idx + 0.5) * KEY_SIZE + idx * KEY_GAP))
			scene.blit(key_lbl, lbl_rect)

	# Draw status label
	status_font = pg.font.SysFont('consolas', 18)

	if not status:
		status_lbl = status_font.render(f'Attempt {attempt_num + 1}/{MAX_ATTEMPTS}', True, (224, 224, 224))
	else:
		status_lbl = status_font.render(status, True, (255, 255, 255))

	j = WORD_LEN * (CELL_SIZE + GAP) + 2 * GRID_OFFSET
	lbl_rect = status_lbl.get_rect(center=(j / 2, 675))
	scene.blit(status_lbl, lbl_rect)

	pg.display.update()


if __name__ == '__main__':
	with open('five_letter_words.txt', 'r') as file:
		all_words = file.read().splitlines()
	pattern_dict = generate_pattern_dict(all_words)

	pg.init()
	pg.display.set_caption('Wordle')
	scene = pg.display.set_mode((WORD_LEN * (CELL_SIZE + GAP) + 2 * GRID_OFFSET,
		MAX_ATTEMPTS * (CELL_SIZE + GAP) + 2 * GRID_OFFSET + 180))

	draw_grid(colour_current_row=False)

	# Generate all possible patterns (3^5 = 243) of green/yellow/grey info
	# i.e. (0, 0, 0, 0, 0) ... (2, 2, 2, 2, 2)
	all_patterns = []
	generate_all_info_patterns(all_patterns)

	words_left = set(all_words)

	# If using assistance to solve online wordle...

	choice = input('\nSolving online wordle? (Y/N)\n>>> ').upper()
	if choice == 'Y':
		# Suggest start word with highest entropy
		entropies = calculate_word_entropies(words_left, pattern_dict, all_patterns)
		attempt_word = max(entropies, key=entropies.get)

		i = 1
		while i <= MAX_ATTEMPTS:
			print(f"\nAttempt {i}/{MAX_ATTEMPTS}: try '{attempt_word}'")
			info = input(
				f"Enter the info (0/1/2 for grey/yellow/green) for '{attempt_word}' (or W if won)\n>>> "
			).upper()

			if info == 'W':
				break

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
			match event.type:
				case pg.QUIT:
					sys.exit()
				case pg.KEYDOWN:
					if game_over and event.key != pg.K_SPACE:
						continue

					match event.key:
						case pg.K_SPACE:  # Reset if game over
							if game_over:
								attempts = [[''] * WORD_LEN for _ in range(MAX_ATTEMPTS)]
								attempt_num = col_num = 0
								green_letters, yellow_letters, grey_letters = [], [], []
								game_over = False
								target_word = np.random.choice(all_words)
								words_left = set(all_words)
								draw_grid(colour_current_row=False)
						case event.key if 97 <= event.key <= 122:  # a - z
							if attempt_num < MAX_ATTEMPTS and col_num < WORD_LEN:
								attempts[attempt_num][col_num] = chr(event.key - 32)  # -32 to capitalise
								col_num += 1
								draw_grid(colour_current_row=False)
						case pg.K_RETURN:  # Submit attempt
							if col_num != WORD_LEN:
								continue

							user_word = ''.join(attempts[attempt_num])

							if user_word not in all_words:
								draw_grid(colour_current_row=False, status=f"'{user_word}' not in word list!")
								pg.time.delay(1500)
								attempts[attempt_num] = [''] * WORD_LEN
								col_num = 0
								draw_grid(colour_current_row=False)
							elif user_word == target_word:
								draw_grid(colour_current_row=True, status='You win!')
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
						case pg.K_BACKSPACE:  # Delete last char
							col_num = max(col_num - 1, 0)
							attempts[attempt_num][col_num] = ''
							draw_grid(colour_current_row=False)
						case pg.K_TAB:  # Use assistance
							# Suggest word left with highest entropy
							entropies = calculate_word_entropies(words_left, pattern_dict, all_patterns)
							attempt_word = max(entropies, key=entropies.get)

							draw_grid(colour_current_row=False, status=f"Try '{attempt_word}'...")
							pg.time.delay(1000)
							draw_grid(colour_current_row=False)
