"""
Steganography demo (LSB text and image steganography)

Author: Sam Barba
Created 01/10/2021
"""

from math import ceil
from PIL import Image
import random

CHARS = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)] + [str(i) for i in range(10)]
IMG = Image.open('img.jpg')
WIDTH, HEIGHT = IMG.size

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def hide_message_in_text(bin_msg):
	container_text = [random.choice(CHARS) for _ in range(len(bin_msg))]

	print('\nContainer text:', ''.join(container_text))

	for idx, c in enumerate(bin_msg):
		n = ord(container_text[idx])
		n = set_bit(n, 0, int(c))
		container_text[idx] = chr(n)

	# Replace special characters in new text with characters that have same LSB
	has_special_chars = any(c not in CHARS for c in container_text)
	if has_special_chars:
		chars_ending_0 = [c for c in CHARS if ord(c) % 2 == 0]
		chars_ending_1 = [c for c in CHARS if ord(c) % 2 == 1]

		for idx, c in enumerate(container_text):
			if c not in CHARS:
				container_text[idx] = random.choice(chars_ending_0 if ord(c) % 2 == 0 else chars_ending_1)

	return ''.join(container_text)

def get_message_from_text(steg_text):
	# Least significant bit of each char
	binary = [str(ord(c) & 1) for c in steg_text]

	chunks = [binary[i:i + 8] for i in range(0, len(binary), 8)]
	msg = [chr(int(''.join(c), 2)) for c in chunks]

	return ''.join(msg)

def hide_message_in_image(bin_msg):
	pixels = IMG.getdata()
	pixels_needed = ceil(len(bin_msg) / 3)

	if pixels_needed > len(pixels):
		raise ValueError('Not enough pixels in image')

	steg_img = IMG.copy()
	msg_idx = 0

	for i in range(pixels_needed):
		pixel = list(pixels[i])

		for j in range(3):
			if msg_idx < len(bin_msg):
				pixel[j] = set_bit(pixel[j], 0, int(bin_msg[msg_idx]))
				msg_idx += 1

		x, y = i % WIDTH, i // WIDTH
		steg_img.putpixel((x, y), tuple(pixel))

	return steg_img

def get_message_from_image(steg_img):
	# Least significant bit of each RGB value of each pixel
	pixels = steg_img.getdata()
	binary = [(pixels[y][x] & 1) for y in range(len(pixels)) for x in range(3)]
	binary = [str(b) for b in binary]

	chunks = [binary[i:i + 8] for i in range(0, len(binary), 8)]
	msg = [chr(int(''.join(c), 2)) for c in chunks]

	return ''.join(msg[:100]) + ' (...)'

def set_bit(n, idx, b):
	"""Set (idx)th bit of number 'n' to 'b'"""

	mask = 1 << idx
	n &= ~mask
	return n | mask if b == 1 else n

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	msg = input('Enter message to hide: ')
	bin_msg = ''.join([format(ord(c), '08b') for c in msg])

	print('\nIn binary:', bin_msg)

	steg_text = hide_message_in_text(bin_msg)

	print('\nMessage hidden in text:', steg_text)
	print(f'\nReading hidden message from text:\n{get_message_from_text(steg_text)}')

	steg_img = hide_message_in_image(bin_msg)
	steg_img.show()

	print(f'\nReading hidden message from image:\n{get_message_from_image(steg_img)}')
