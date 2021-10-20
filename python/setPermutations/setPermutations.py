# Permutations demo
# Author: Sam Barba
# Created 08/09/2021

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Heap's algorithm for generating all permutations of n objects
def permutations(n, charSet, results):
	if n == 1:
		results.append("".join(charSet))
	else:
		for i in range(n):
			permutations(n - 1, charSet, results)
			if n % 2 == 0:
				charSet[i], charSet[n - 1] = charSet[n - 1], charSet[i]
			else:
				charSet[0], charSet[n - 1] = charSet[n - 1], charSet[0]

def permutationsWithRepetitionLengthK(charSet, k, permutationRepetitionResults, prefix=""):
	if k == 0:
		permutationRepetitionResults.append(prefix)
	else:
		for c in charSet:
			newPrefix = prefix + c
			permutationsWithRepetitionLengthK(charSet, k - 1, permutationRepetitionResults, newPrefix)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

charSet = input("Enter a word: ").upper()

permutationResults, permutationRepetitionResults = [], []
permutations(len(charSet), list(charSet), permutationResults)
permutationsWithRepetitionLengthK(list(set(charSet)), len(charSet), permutationRepetitionResults)

permutationResults = sorted(list(set(permutationResults)))
permutationRepetitionResults.sort()

print(f"\n{len(permutationResults)} unique permutations (no repetition) of '{charSet}':\n{permutationResults}")
print(f"\n{len(permutationRepetitionResults)} permutations (with repetition) of '{charSet}':\n{permutationRepetitionResults}")
