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

def permutationsWithRepetitionLengthK(charSet, k, permutationRepetitionResults, prefix = ""):
	if k == 0:
		permutationRepetitionResults.append(prefix)
	else:
		for c in charSet:
			newPrefix = prefix + c
			permutationsWithRepetitionLengthK(charSet, k - 1, permutationRepetitionResults, newPrefix)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

while True:
	charSet = input("Enter a word: ").upper()

	permutationResults, permutationRepetitionResults = [], []
	permutations(len(charSet), list(charSet), permutationResults)
	permutationsWithRepetitionLengthK(list(set(charSet)), len(charSet), permutationRepetitionResults)

	permutationResults = sorted(list(set(permutationResults)))
	permutationRepetitionResults.sort()

	print("\n{} unique permutations (no repetition) of '{}':\n{}".format(len(permutationResults), charSet, permutationResults))
	print("\n{} permutations (with repetition) of '{}':\n{}".format(len(permutationRepetitionResults), charSet, permutationRepetitionResults))

	choice = input("\nEnter to continue or X to exit: ").upper()
	if len(choice) > 0 and choice[0] == 'X':
		break
	print()
