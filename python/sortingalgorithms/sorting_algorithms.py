"""
Demo of some different sorting algorithms

Author: Sam Barba
Created 06/09/2021
"""

from time import perf_counter

import numpy as np

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def bubble_sort(arr):
	swap = True

	while swap:
		swap = False

		for idx, n in enumerate(arr[:-1]):
			if n > arr[idx + 1]:
				arr[idx], arr[idx + 1] = arr[idx + 1], n
				swap = True

def cocktail_shaker_sort(arr):
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

def comb_sort(arr):
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

def counting_sort(arr):
	counts = [0] * (max(arr) + 1)

	for i in arr:
		counts[i] += 1

	output = []
	for idx, value in enumerate(counts):
		output.extend([idx] * value)

	arr[:] = output

def insertion_sort(arr):
	for idx, n in enumerate(arr[1:]):
		j = idx
		while j >= 0 and arr[j] > n:
			arr[j + 1] = arr[j]
			j -= 1
		arr[j + 1] = n

def merge_sort(arr):
	if len(arr) <= 1: return

	mid = len(arr) // 2
	l, r = arr[:mid], arr[mid:]

	merge_sort(l)  # Sort copy of first half
	merge_sort(r)  # Sort copy of second half

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

def radix_sort(arr):  # Least Significant Digit
	max_digits = len(str(max(arr)))

	for i in range(max_digits):
		buckets = [[] for _ in range(10)]

		for n in arr:
			idx = (n // (10 ** i)) % 10
			buckets[idx].append(n)

		arr[:] = [n for b in buckets for n in b]

def selection_sort(arr):
	for idx, n in enumerate(arr[:-1]):
		min_idx = idx
		for j in range(idx + 1, len(arr)):
			if arr[j] < arr[min_idx]:
				min_idx = j
		arr[idx], arr[min_idx] = arr[min_idx], n

def shell_sort(arr):
	gap = len(arr) // 2

	while gap:
		for idx, n in enumerate(arr[gap:], start=gap):
			while idx >= gap and arr[idx - gap] > n:
				arr[idx] = arr[idx - gap]
				idx -= gap
			arr[idx] = n
		gap = 1 if gap == 2 else (gap * 5) // 11

def test_function(sort_func, arr):
	print('Sorting with {:.<28}'.format("'" + sort_func.__name__ + "'"), end='')
	start = perf_counter()
	sort_func(arr)
	interval = perf_counter() - start
	print(f' done in {(1000 * interval):.0f} ms')

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	max_num = 10_000

	nums = np.random.randint(0, max_num, size=max_num)

	test_function(sorted, nums[:])
	test_function(bubble_sort, nums[:])
	test_function(cocktail_shaker_sort, nums[:])
	test_function(comb_sort, nums[:])
	test_function(counting_sort, nums[:])
	test_function(insertion_sort, nums[:])
	test_function(merge_sort, nums[:])
	test_function(quicksort, nums[:])
	test_function(radix_sort, nums[:])
	test_function(selection_sort, nums[:])
	test_function(shell_sort, nums[:])
