# Steganography demo (LSB text and image steganography)
# Author: Sam Barba
# Created 01/10/2021

from math import ceil
from PIL import Image
import random

CHARS = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)] + [str(i) for i in range(10)]
IMG = Image.open("img.jpg")
WIDTH, HEIGHT = IMG.size

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def hideMessageInText(binMsg):
	containerText = [random.choice(CHARS) for i in range(len(binMsg))]

	print("\nContainer text:", "".join(containerText))

	for i in range(len(binMsg)):
		n = ord(containerText[i])
		n = setBit(n, 0, int(binMsg[i]))
		containerText[i] = chr(n)

	# Replace special characters in new text with characters that have same LSB
	hasSpecialChars = not all(c in CHARS for c in containerText)
	if hasSpecialChars:
		charsEnding0 = [c for c in CHARS if ord(c) % 2 == 0]
		charsEnding1 = [c for c in CHARS if ord(c) % 2 == 1]

		for idx, c in enumerate(containerText):
			if c not in CHARS:
				containerText[idx] = random.choice(charsEnding0 if ord(c) % 2 == 0 else charsEnding1)

	return "".join(containerText)

def getMessageFromText(stegText):
	# Least significant bit of each char
	binary = [str(ord(c) & 1) for c in stegText]

	chunks = [binary[i:i + 8] for i in range(0, len(binary), 8)]
	msg = [chr(int("".join(c), 2)) for c in chunks]

	return "".join(msg)

def hideMessageInImage(binMsg):
	pixels = IMG.getdata()
	pixelsNeeded = ceil(len(binMsg) / 3)

	if pixelsNeeded > len(pixels):
		raise ValueError("Not enough pixels in image")

	stegImg = IMG.copy()
	msgIdx = 0

	for i in range(pixelsNeeded):
		pixel = list(pixels[i])

		for j in range(3):
			if msgIdx < len(binMsg):
				pixel[j] = setBit(pixel[j], 0, int(binMsg[msgIdx]))
				msgIdx += 1

		x, y = i % WIDTH, i // WIDTH
		stegImg.putpixel((x, y), tuple(pixel))

	return stegImg

def getMessageFromImage(stegImg):
	# Least significant bit of each RGB value of each pixel
	pixels = stegImg.getdata()
	binary = [(pixels[y][x] & 1) for y in range(len(pixels)) for x in range(3)]
	binary = [str(b) for b in binary]

	chunks = [binary[i:i + 8] for i in range(0, len(binary), 8)]
	msg = [chr(int("".join(c), 2)) for c in chunks]

	return "".join(msg[:100]) + " (...)"

# Set idx:th bit of number 'n' to 'b'
def setBit(n, idx, b):
	mask = 1 << idx
	n &= ~mask
	return n | mask if b == 1 else n

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

while True:
	msg = input("Enter message to hide: ")
	binMsg = "".join([format(ord(c), "08b") for c in msg])

	print("\nIn binary:", binMsg)

	stegText = hideMessageInText(binMsg)

	print("\nMessage hidden in text:", stegText)
	print("\nReading hidden message from text:\n{}".format(getMessageFromText(stegText)))

	stegImg = hideMessageInImage(binMsg)
	stegImg.show()

	print("\nReading hidden message from image:\n{}".format(getMessageFromImage(stegImg)))

	choice = input("\nEnter to continue or X to exit: ").upper()
	if len(choice) > 0 and choice[0] == 'X':
		break
	print()
