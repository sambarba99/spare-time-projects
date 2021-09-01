# Roman Numeral Converter
# Author: Sam Barba
# Created 06/09/2021

ALL_NUMERAL_VALS = {"M": 1000,
	"CM": 900,
	"D": 500,
	"CD": 400,
	"C": 100,
	"XC": 90,
	"L": 50,
	"XL": 40,
	"X": 10,
	"IX": 9,
	"V": 5,
	"IV": 4,
	"I": 1}

SINGLE_NUMERAL_VALS = {"I": 1,
	"V": 5,
	"X": 10,
	"L": 50,
	"C": 100,
	"D": 500,
	"M": 1000}

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def intToNumerals(n):
	if n <= 0: return str(n)

	numerals = ""
	for k, v in ALL_NUMERAL_VALS.items():
		while n >= v:
			numerals += k
			n -= v

	return numerals

def numeralsToInt(numerals):
	n = 0

	for i in range(len(numerals)):
		val = SINGLE_NUMERAL_VALS[numerals[i]]

		if i + 1 < len(numerals) and SINGLE_NUMERAL_VALS[numerals[i + 1]] > val:
			n -= val
		else:
			n += val

	return n

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

while True:
	choice = int(input("Enter 1 to convert to numerals or 2 to convert from numerals: "))
	print()

	if choice == 1:
		n = int(input("Enter the number to convert: "))
		print("\nResult:", intToNumerals(n))
	elif choice == 2:
		numerals = input("Enter the numerals to convert: ").upper()
		print("\nResult:", numeralsToInt(numerals))

	choice = input("\nEnter to continue or X to exit: ").upper()
	if len(choice) > 0 and choice[0] == 'X':
		break
	print()
