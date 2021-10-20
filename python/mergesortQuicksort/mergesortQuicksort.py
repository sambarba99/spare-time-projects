# Mergesort and Quicksort demo
# Author: Sam Barba
# Created 06/09/2021

import random
import time

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def mergesort(arr):
	if len(arr) <= 1: return

	mid = len(arr) // 2
	l, r = arr[:mid], arr[mid:]

	mergesort(l) # Sort copy of first half
	mergesort(r) # Sort copy of second half

	merge(l, r, arr) # Merge sorted halves back into arr

def merge(l, r, arr):
	i = j = 0
	while i + j < len(arr):
		if j == len(r) or (i < len(l) and l[i] < r[j]):
			arr[i + j] = l[i]
			i += 1
		else:
			arr[i + j] = r[j]
			j += 1

def quicksort(arr):
	if len(arr) <= 1: return

	less, equal, greater = [], [], []

	pivot = arr[len(arr) // 2]
	for x in arr:
		if x < pivot: less.append(x)
		elif x == pivot: equal.append(x)
		else: greater.append(x)

	quicksort(less)
	quicksort(greater)
	arr[:] = less + equal + greater

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

size = 10 ** 6

nums = [random.randrange(size) for _ in range(size)]
numsCopy = nums[:]

print("Sorting with mergesort...")
start = time.perf_counter()
mergesort(nums)
end = time.perf_counter()
timeTaken = round((end - start) * 1000)
print(f"Sorted in {timeTaken} ms")

print("\nSorting with quicksort...")
start = time.perf_counter()
quicksort(numsCopy)
end = time.perf_counter()
timeTaken = round((end - start) * 1000)
print(f"Sorted in {timeTaken} ms")
