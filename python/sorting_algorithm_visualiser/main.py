"""
Sorting algorithm visualisation

Author: Sam Barba
Created 10/01/2024
"""

import sys

import numpy as np
import pygame as pg


WIDTH = 1800
HEIGHT = 900


def bubble_sort(arr):
	for i in range(array_size - 1):
		for j in range(array_size - i - 1):
			if arr[j] > arr[j + 1]:
				arr[j], arr[j + 1] = arr[j + 1], arr[j]
				draw_array_general(arr, highlight_indices=[j, j + 1])


def cocktail_shaker_sort(arr):
	swapped = True
	start, end = 0, array_size - 1

	while swapped:
		for i in range(start, end):
			if arr[i] > arr[i + 1]:
				arr[i], arr[i + 1] = arr[i + 1], arr[i]
				draw_array_general(arr, highlight_indices=[i, i + 1])
				swapped = True

		if not swapped: break

		swapped = False
		end -= 1

		for i in range(end - 1, start - 1, -1):
			if arr[i] > arr[i + 1]:
				arr[i], arr[i + 1] = arr[i + 1], arr[i]
				draw_array_general(arr, highlight_indices=[i, i + 1])
				swapped = True

		start += 1


def comb_sort(arr):
	gap = array_size
	k = 1.3
	done = False

	while not done:
		gap = int(gap / k)
		if gap <= 1:
			gap = 1
			done = True

		i = 0
		while i + gap < array_size:
			if arr[i] > arr[i + gap]:
				arr[i], arr[i + gap] = arr[i + gap], arr[i]
				draw_array_general(arr, highlight_indices=[i, i + gap])
				done = False
			i += 1


def heap_sort(arr):
	def heapify(n, i):
		largest = i
		left_child = 2 * i + 1
		right_child = 2 * i + 2

		if left_child < n and arr[left_child] > arr[largest]:
			largest = left_child
		if right_child < n and arr[right_child] > arr[largest]:
			largest = right_child
		if largest != i:
			arr[i], arr[largest] = arr[largest], arr[i]
			draw_array_general(arr, highlight_indices=[i, largest])
			heapify(n, largest)


	# Build a max heap
	for i in range(array_size // 2 - 1, -1, -1):
		heapify(array_size, i)

	# Extract elements one by one
	for i in range(array_size - 1, 0, -1):
		arr[i], arr[0] = arr[0], arr[i]
		draw_array_general(arr, highlight_indices=[i, 0])
		heapify(i, 0)


def insertion_sort(arr):
	for i in range(1, array_size):
		key = arr[i]
		j = i - 1
		while j >= 0 and arr[j] > key:
			arr[j + 1] = arr[j]
			draw_array_general(arr, highlight_indices=[j + 1])
			j -= 1
		arr[j + 1] = key
		draw_array_general(arr, highlight_indices=[j + 1])


def merge(arr, lo, mid, hi):
	len1 = mid - lo + 1
	len2 = hi - mid

	left = [arr[lo + i] for i in range(len1)]
	right = [arr[mid + i + 1] for i in range(len2)]

	i, j, k = 0, 0, lo

	while i < len1 and j < len2:
		if left[i] <= right[j]:
			arr[k] = left[i]
			i += 1
		else:
			arr[k] = right[j]
			j += 1
		draw_array_general(arr, highlight_indices=[k])
		k += 1

	while i < len1:
		arr[k] = left[i]
		draw_array_general(arr, highlight_indices=[k])
		i += 1
		k += 1
	while j < len2:
		arr[k] = right[j]
		draw_array_general(arr, highlight_indices=[k])
		j += 1
		k += 1


def merge_sort(arr):
	"""Non-recursive implementation"""

	size = 1

	while size < array_size:
		for lo in range(0, array_size, 2 * size):
			mid = min(lo + size - 1, array_size - 1)
			hi = min(lo + 2 * size - 1, array_size - 1)
			if mid < hi:
				merge(arr, lo, mid, hi)
		size *= 2


def quicksort_hoare_partition(arr, lo=None, hi=None):
	def partition():
		pivot_idx = (lo + hi) // 2
		pivot = arr[pivot_idx]
		i, j = lo - 1, hi + 1

		while True:
			i += 1
			while arr[i] < pivot:
				i += 1
			j -= 1
			while arr[j] > pivot:
				j -= 1

			if i >= j:
				return j

			arr[i], arr[j] = arr[j], arr[i]
			draw_array_quicksort(arr, lo, hi, pivot_idx, i, j)


	if lo is None and hi is None:  # First call
		lo, hi = 0, array_size - 1

	if lo < hi:
		partition_idx = partition()
		quicksort_hoare_partition(arr, lo, partition_idx)
		quicksort_hoare_partition(arr, partition_idx + 1, hi)


def quicksort_3_way_partition(arr, lo=None, hi=None):
	def partition():
		pivot = arr[lo]
		less_than, greater_than = lo, hi
		i = lo + 1

		while i <= greater_than:
			if arr[i] < pivot:
				arr[i], arr[less_than] = arr[less_than], arr[i]
				i += 1
				less_than += 1
				draw_array_quicksort(arr, lo, hi, None, less_than, greater_than)
			elif arr[i] > pivot:
				arr[i], arr[greater_than] = arr[greater_than], arr[i]
				greater_than -= 1
				draw_array_quicksort(arr, lo, hi, None, less_than, greater_than)
			else:
				i += 1

		return less_than, greater_than


	if lo is None and hi is None:  # First call
		lo, hi = 0, array_size - 1

	if lo < hi:
		less_than, greater_than = partition()
		quicksort_3_way_partition(arr, lo, less_than - 1)
		quicksort_3_way_partition(arr, greater_than + 1, hi)


def selection_sort(arr):
	for i in range(array_size - 1):
		min_idx = i
		for j in range(i + 1, array_size):
			if arr[j] < arr[min_idx]:
				min_idx = j
		arr[i], arr[min_idx] = arr[min_idx], arr[i]
		draw_array_general(arr, highlight_indices=[i, min_idx])


def shell_sort(arr):
	gap = array_size // 2

	while gap:
		for i in range(gap, array_size):
			key = arr[i]
			while i >= gap and arr[i - gap] > key:
				arr[i] = arr[i - gap]
				draw_array_general(arr, highlight_indices=[i])
				i -= gap
			arr[i] = key
			draw_array_general(arr, highlight_indices=[i])
		gap = 1 if gap == 2 else int(gap * 5 / 11)


def tim_sort(arr, min_merge=32):
	def calc_min_run(n):
		r = 0
		while n >= min_merge:
			r |= n & 1
			n //= 2
		return n + r

	def tim_sort_insertion(lo, hi):
		for i in range(lo + 1, hi + 1):
			j = i
			while j > lo and arr[j] < arr[j - 1]:
				arr[j], arr[j - 1] = arr[j - 1], arr[j]
				draw_array_general(arr, highlight_indices=[j, j - 1])
				j -= 1


	run = calc_min_run(array_size)

	for start in range(0, array_size, run):
		end = min(start + run - 1, array_size - 1)
		tim_sort_insertion(start, end)

	size = run

	while size < array_size:
		for lo in range(0, array_size, 2 * size):
			mid = min(lo + size - 1, array_size - 1)
			hi = min(lo + 2 * size - 1, array_size - 1)
			if mid < hi:
				merge(arr, lo, mid, hi)
		size *= 2


def draw_array_general(arr, colour='blue', highlight_indices=None, shuffle=False):
	for event in pg.event.get():
		if event.type == pg.QUIT:
			sys.exit()

	scene.fill('black')

	for i, bar_h in enumerate(arr):
		col = 'white' if highlight_indices and (i in highlight_indices) else colour
		pg.draw.rect(scene, col, (i * bar_w, HEIGHT - bar_h, bar_w, bar_h))

	caption_lbl = font.render('Shuffling...' if shuffle else caption, True, (220, 220, 220))
	lbl_rect = caption_lbl.get_rect(center=(WIDTH // 2, 20))
	scene.blit(caption_lbl, lbl_rect)
	pg.display.update()


def draw_array_quicksort(arr, lo, hi, pivot_idx, swap_i, swap_j):
	for event in pg.event.get():
		if event.type == pg.QUIT:
			sys.exit()

	scene.fill('black')

	for i, bar_h in enumerate(arr):
		if i in (lo, hi): colour = '#ff8000'
		elif i == pivot_idx and swap_i <= pivot_idx <= swap_j: colour = 'green'
		elif i in (swap_i, swap_j): colour = 'white'
		else: colour = 'blue'

		pg.draw.rect(scene, colour, (i * bar_w, HEIGHT - bar_h, bar_w, bar_h))

	caption_lbl = font.render(caption, True, (220, 220, 220))
	lbl_rect = caption_lbl.get_rect(center=(WIDTH // 2, 20))
	scene.blit(caption_lbl, lbl_rect)
	pg.display.update()


if __name__ == '__main__':
	funcs = [
		(bubble_sort, 'Bubble sort', 150),
		(cocktail_shaker_sort, 'Cocktail shaker sort', 150),
		(comb_sort, 'Comb sort', 900),
		(heap_sort, 'Heap sort', 600),
		(insertion_sort, 'Insertion sort', 150),
		(merge_sort, 'Merge sort', 900),
		(quicksort_hoare_partition, 'Quicksort (Hoare partitioning)', 1800),
		(quicksort_3_way_partition, 'Quicksort (3-way partitioning)', 900),
		(selection_sort, 'Selection sort', 1800),
		(shell_sort, 'Shell sort', 600),
		(tim_sort, 'Timsort', 900)
	]

	pg.init()
	pg.display.set_caption('Sorting algorithm visualiser')
	scene = pg.display.set_mode((WIDTH, HEIGHT))
	font = pg.font.SysFont('consolas', 16)

	for func, func_name, array_size in funcs:
		bar_w = WIDTH // array_size
		caption = f'{func_name} ({array_size} elements)'

		linspace = np.linspace(10, HEIGHT - 30, num=array_size, dtype=int)
		array = list(linspace)
		draw_array_general(array, shuffle=True)
		pg.time.delay(1000)
		for i in range(array_size - 1):
			rand_idx = np.random.randint(i + 1, array_size)
			draw_array_general(array, highlight_indices=[i, rand_idx], shuffle=True)
			array[i], array[rand_idx] = array[rand_idx], array[i]

		draw_array_general(array)
		pg.time.delay(2000)
		func(array)
		draw_array_general(array, colour='green')  # Draw final sorted array in green
		pg.time.delay(2000)
