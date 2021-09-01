# Number Base Converter
# Author: Sam Barba
# Created 04/09/2021

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def toDecimalFromBase(numStr, fromBase):
	decNum = 0
	power = fromBase ** (len(numStr) - 1)

	for n in numStr:
		val = ord(n) - ord('0') if '0' <= n <= '9' else ord(n) - ord('A') + 10
		decNum += val * power
		power //= fromBase

	return decNum

def toBaseFromDecimal(decNum, toBase):
	remainders = []

	while decNum > 0:
		remainder = decNum % toBase
		remainder = str(remainder) if remainder < 10 else str(chr(55 + remainder))
		remainders.append(remainder)
		decNum //= toBase

	return "".join(remainders[::-1])

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

while True:
	numStr = input("Enter a number: ").upper()
	base = int(input("Enter its base: "))
	print()

	decNum = int(numStr) if base == 10 else toDecimalFromBase(numStr, base)

	for i in range(2, 17):
		if i == base: continue

		numInBaseI = toBaseFromDecimal(decNum, i)
		print("{} from base {} to base {}: {}".format(numStr, base, i, numInBaseI))

	choice = input("\nEnter to continue or X to exit: ").upper()
	if len(choice) > 0 and choice[0] == 'X':
		break
	print()
