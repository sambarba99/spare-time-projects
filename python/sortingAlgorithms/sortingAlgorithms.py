# Demo of some different sorting algorithms
# Author: Sam Barba
# Created 06/09/2021

import random
from time import perf_counter

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def bubbleSort(arr):
	swap = True

	while swap:
		swap = False

		for idx, n in enumerate(arr[:-1]):
			if n > arr[idx + 1]:
				arr[idx], arr[idx + 1] = arr[idx + 1], n
				swap = True

def cocktailShakerSort(arr):
	swap = True

	while swap:
		for idx, n in enumerate(arr[:-1]):
			if n > arr[idx + 1]:
				arr[idx], arr[idx + 1] = arr[idx + 1], n
				swap = True

		if not swap: break

		swap = False

		for idx, n in reversed(list(enumerate(arr[:-1]))):
			if n > arr[idx + 1]:
				arr[idx], arr[idx + 1] = arr[idx + 1], n
				swap = True

def combSort(arr):
	gap = len(arr)
	k = 1.3
	done = False

	while not done:
		gap = int(gap / k)
		if gap <= 1:
			gap = 1
			done = True

		i = 0
		while i + gap < len(arr):
			if arr[i] > arr[i + gap]:
				arr[i], arr[i + gap] = arr[i + gap], arr[i]
				done = False
			i += 1

def countingSort(arr):
	counts = [0] * (max(arr) + 1)

	for i in arr:
		counts[i] += 1

	output = []
	for idx, value in enumerate(counts):
		output.extend([idx] * value)

	arr[:] = output

def insertionSort(arr):
	for idx, n in enumerate(arr[1:], start=1):
		j = idx - 1
		while j >= 0 and arr[j] > n:
			arr[j + 1] = arr[j]
			j -= 1
		arr[j + 1] = n

def mergesort(arr):
	if len(arr) <= 1: return

	mid = len(arr) // 2
	l, r = arr[:mid], arr[mid:]

	mergesort(l) # Sort copy of first half
	mergesort(r) # Sort copy of second half

	# Merge sorted halves back into arr
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

	pivot = arr[len(arr) // 2]
	less = [i for i in arr if i < pivot]
	equal = [i for i in arr if i == pivot]
	greater = [i for i in arr if i > pivot]

	quicksort(less)
	quicksort(greater)
	arr[:] = less + equal + greater

def radixSort(arr): # Least Significant Digit
	maxDigits = len(str(max(arr)))

	for i in range(maxDigits):
		buckets = [[] for _ in range(10)]

		for n in arr:
			idx = (n // (10 ** i)) % 10
			buckets[idx].append(n)

		arr[:] = sum(buckets, start=[]) # Flatten buckets list

def selectionSort(arr):
	for idx, n in enumerate(arr[:-1]):
		minIdx = idx
		for j in range(idx + 1, len(arr)):
			if arr[j] < arr[minIdx]:
				minIdx = j
		arr[idx], arr[minIdx] = arr[minIdx], n

def shellSort(arr):
	gap = len(arr) // 2

	while gap:
		for idx, n in enumerate(arr[gap:], start=gap):
			while idx >= gap and arr[idx - gap] > n:
				arr[idx] = arr[idx - gap]
				idx -= gap
			arr[idx] = n
		gap = 1 if gap == 2 else (gap * 5) // 11

def testFunction(sortFunc, arr):
	print("Sorting with {:.<25}".format("'" + sortFunc.__name__ + "'"), end="")
	start = perf_counter()
	sortFunc(arr)
	end = perf_counter()
	timeTaken = round((end - start) * 1000)
	print(f" done in {timeTaken} ms")

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

size = 10 ** 4

nums = [random.randrange(size) for _ in range(size)]

testFunction(sorted, nums[:])
testFunction(bubbleSort, nums[:])
testFunction(cocktailShakerSort, nums[:])
testFunction(combSort, nums[:])
testFunction(countingSort, nums[:])
testFunction(insertionSort, nums[:])
testFunction(mergesort, nums[:])
testFunction(quicksort, nums[:])
testFunction(radixSort, nums[:])
testFunction(selectionSort, nums[:])
testFunction(shellSort, nums[:])
