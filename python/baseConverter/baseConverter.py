# Number Base Converter
# Author: Sam Barba
# Created 04/09/2021

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def to_decimal_from_base(num_str, from_base):
	dec_num = 0
	power = from_base ** (len(num_str) - 1)

	for n in num_str:
		val = ord(n) - ord('0') if '0' <= n <= '9' else ord(n) - ord('A') + 10
		dec_num += val * power
		power //= from_base

	return dec_num

def to_base_from_decimal(dec_num, to_base):
	remainders = []

	while dec_num:
		remainder = dec_num % to_base
		remainder = str(remainder) if remainder < 10 else str(chr(55 + remainder))
		remainders.append(remainder)
		dec_num //= to_base

	return "".join(remainders[::-1])

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

input_num = input("Enter a number: ").upper()
base = int(input("Enter its base: "))
print()

input_num_to_dec = int(input_num) if base == 10 else to_decimal_from_base(input_num, base)

for i in range(2, 17):
	if i == base: continue

	num_in_base_i = to_base_from_decimal(input_num_to_dec, i)
	print(f"{input_num} from base {base} to base {i}: {num_in_base_i}")
