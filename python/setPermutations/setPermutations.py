# Permutations demo
# Author: Sam Barba
# Created 08/09/2021

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Heap's algorithm for generating all permutations of n objects
def permutations(n, char_set, results):
	if n == 1:
		results.append("".join(char_set))
	else:
		for i in range(n):
			permutations(n - 1, char_set, results)
			if n % 2 == 0:
				char_set[i], char_set[n - 1] = char_set[n - 1], char_set[i]
			else:
				char_set[0], char_set[n - 1] = char_set[n - 1], char_set[0]

def permutations_with_repetition_length_k(char_set, k, permutation_repetition_results, prefix=""):
	if k == 0:
		permutation_repetition_results.append(prefix)
	else:
		for c in char_set:
			new_prefix = prefix + c
			permutations_with_repetition_length_k(char_set, k - 1, permutation_repetition_results, new_prefix)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

char_set = input("Enter a word: ").upper()

permutation_results, permutation_repetition_results = [], []
permutations(len(char_set), list(char_set), permutation_results)
permutations_with_repetition_length_k(list(set(char_set)), len(char_set), permutation_repetition_results)

permutation_results = sorted(list(set(permutation_results)))
permutation_repetition_results.sort()

print(f"\n{len(permutation_results)} unique permutations (no repetition) of '{char_set}':\n{permutation_results}")
print(f"\n{len(permutation_repetition_results)} permutations (with repetition) of '{char_set}':\n{permutation_repetition_results}")
