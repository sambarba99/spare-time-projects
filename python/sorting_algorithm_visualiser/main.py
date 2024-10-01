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


def swap(arr, i, j):
	arr[i], arr[j] = arr[j], arr[i]
	draw_array(arr, index_colours={i: 'white', j: 'white'})


def bubble_sort(arr):
	for i in range(array_size - 1):
		for j in range(array_size - i - 1):
			if arr[j] > arr[j + 1]:
				swap(arr, j, j + 1)


def cocktail_shaker_sort(arr):
	start, end = 0, array_size - 1
	done = False

	while not done:
		done = True

		for i in range(start, end):
			if arr[i] > arr[i + 1]:
				swap(arr, i, i + 1)
				done = False

		if done:
			break

		end -= 1

		for i in range(end - 1, start - 1, -1):
			if arr[i] > arr[i + 1]:
				swap(arr, i, i + 1)
				done = False

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
				swap(arr, i, i + gap)
				done = False
			i += 1


def cycle_sort(arr):
	for cycle_start in range(array_size - 1):
		item = arr[cycle_start]
		pos = cycle_start
		for i in range(cycle_start + 1, array_size):
			if arr[i] < item:
				pos += 1

		if pos == cycle_start:
			continue

		while item == arr[pos]:
			pos += 1

		arr[pos], item = item, arr[pos]
		draw_array(arr, index_colours={cycle_start: 'white', pos: 'white'})
		while pos != cycle_start:
			pos = cycle_start
			for i in range(cycle_start + 1, array_size):
				if arr[i] < item:
					pos += 1

			while item == arr[pos]:
				pos += 1

			arr[pos], item = item, arr[pos]
			draw_array(arr, index_colours={cycle_start: 'white', pos: 'white'})


def double_sort(arr):
	for _ in range(int(((array_size - 1) / 2) + 1)):
		for j in range(array_size - 1):
			if arr[j + 1] < arr[j]:
				swap(arr, j, j + 1)
			if arr[array_size - j - 1] < arr[array_size - j - 2]:
				swap(arr, array_size - j - 1, array_size - j - 2)


def exchange_sort(arr):
	for i in range(array_size):
		for j in range(i + 1, array_size):
			if arr[i] > arr[j]:
				swap(arr, i, j)


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
			swap(arr, i, largest)
			heapify(n, largest)


	# Build a max heap
	for i in range(array_size // 2 - 1, -1, -1):
		heapify(array_size, i)

	# Extract elements one by one
	for i in range(array_size - 1, 0, -1):
		swap(arr, 0, i)
		heapify(i, 0)


def insertion_sort(arr):
	for i in range(1, array_size):
		key = arr[i]
		j = i - 1
		while j >= 0 and arr[j] > key:
			arr[j + 1] = arr[j]
			draw_array(arr, index_colours={j + 1: 'white'})
			j -= 1
		arr[j + 1] = key
		draw_array(arr, index_colours={j + 1: 'white'})


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
		draw_array(arr, index_colours={k: 'white'})
		k += 1

	while i < len1:
		arr[k] = left[i]
		draw_array(arr, index_colours={k: 'white'})
		i += 1
		k += 1
	while j < len2:
		arr[k] = right[j]
		draw_array(arr, index_colours={k: 'white'})
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


def odd_even_sort(arr):
	done = False

	while not done:
		done = True

		for i in range(0, array_size - 1, 2):  # Even indices
			if arr[i] > arr[i + 1]:
				swap(arr, i, i + 1)
				done = False

		for i in range(1, array_size - 1, 2):  # Odd indices
			if arr[i] > arr[i + 1]:
				swap(arr, i, i + 1)
				done = False


def quicksort_hoare_partition(arr):
	"""Non-recursive implementation"""

	def partition(lo, hi):
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
			draw_array(arr, index_colours={lo: '#ff8000', hi: '#ff8000', pivot_idx: 'green', i: 'white', j: 'white'})


	stack = [(0, array_size - 1)]

	while stack:
		lo, hi = min(stack)  # Instead of stack.pop() (for visualisation purposes)
		stack.remove((lo, hi))

		if lo < hi:
			partition_idx = partition(lo, hi)
			stack.append((lo, partition_idx))  # Left sub-array
			stack.append((partition_idx + 1, hi))  # Right sub-array


def quicksort_3_way_partition(arr):
	"""Non-recursive implementation"""

	def partition(lo, hi):
		pivot = arr[lo]
		less_than, greater_than = lo, hi
		i = lo + 1

		while i <= greater_than:
			if arr[i] < pivot:
				arr[i], arr[less_than] = arr[less_than], arr[i]
				i += 1
				less_than += 1
				draw_array(arr, index_colours={lo: '#ff8000', hi: '#ff8000', less_than: 'white', greater_than: 'white'})
			elif arr[i] > pivot:
				arr[i], arr[greater_than] = arr[greater_than], arr[i]
				greater_than -= 1
				draw_array(arr, index_colours={lo: '#ff8000', hi: '#ff8000', less_than: 'white', greater_than: 'white'})
			else:
				i += 1

		return less_than, greater_than


	stack = [(0, array_size - 1)]

	while stack:
		lo, hi = min(stack)  # Instead of stack.pop() (for visualisation purposes)
		stack.remove((lo, hi))

		if lo < hi:
			less_than, greater_than = partition(lo, hi)
			stack.append((lo, less_than - 1))  # Left sub-array
			stack.append((greater_than + 1, hi))  # Right sub-array


def selection_sort(arr):
	for i in range(array_size - 1):
		min_idx = i
		for j in range(i + 1, array_size):
			if arr[j] < arr[min_idx]:
				min_idx = j
		swap(arr, i, min_idx)


def shell_sort(arr):
	gap = array_size // 2

	while gap:
		for i in range(gap, array_size):
			key = arr[i]
			while i >= gap and arr[i - gap] > key:
				arr[i] = arr[i - gap]
				draw_array(arr, index_colours={i: 'white'})
				i -= gap
			arr[i] = key
			draw_array(arr, index_colours={i: 'white'})
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
				swap(arr, j, j - 1)
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


def bogosort(arr):
	while arr != sorted(arr):
		np.random.shuffle(arr)
		draw_array(arr)


def draw_array(arr, default_colour='blue', index_colours=None, shuffling=False):
	if index_colours is None:
		index_colours = dict()

	for event in pg.event.get():
		if event.type == pg.QUIT:
			sys.exit()

	scene.fill('black')

	for i, bar_h in enumerate(arr):
		col = index_colours.get(i, default_colour)
		pg.draw.rect(scene, col, (i * bar_w, HEIGHT - bar_h, bar_w, bar_h))

	header_lbl = font.render('Shuffling...' if shuffling else heading, True, (224, 224, 224))
	lbl_rect = header_lbl.get_rect(center=(WIDTH // 2, 30))
	scene.blit(header_lbl, lbl_rect)
	pg.display.update()


if __name__ == '__main__':
	funcs = [
		(bubble_sort, 'Bubble sort', 150),
		(cocktail_shaker_sort, 'Cocktail shaker sort', 150),
		(comb_sort, 'Comb sort', 900),
		(cycle_sort, 'Cycle sort', 1800),
		(double_sort, 'Double sort', 150),
		(exchange_sort, 'Exchange sort', 150),
		(heap_sort, 'Heap sort', 600),
		(insertion_sort, 'Insertion sort', 150),
		(merge_sort, 'Merge sort', 900),
		(odd_even_sort, 'Odd-even sort', 150),
		(quicksort_hoare_partition, 'Quicksort (Hoare partitioning)', 1800),
		(quicksort_3_way_partition, 'Quicksort (3-way partitioning)', 900),
		(selection_sort, 'Selection sort', 1800),
		(shell_sort, 'Shell sort', 600),
		(tim_sort, 'Timsort', 900),
		(bogosort, 'Bogosort', 6)
	]

	pg.init()
	pg.display.set_caption('Sorting algorithm visualiser')
	scene = pg.display.set_mode((WIDTH, HEIGHT))
	font = pg.font.SysFont('consolas', 20)

	for func, func_name, array_size in funcs:
		bar_w = WIDTH // array_size
		heading = f'{func_name} ({array_size} elements)'

		array = list(np.linspace(10, HEIGHT - 50, array_size, dtype=int))
		draw_array(array, shuffling=True)
		pg.time.delay(1000)
		for i in range(array_size - 1):
			rand_idx = np.random.randint(i + 1, array_size)
			draw_array(array, index_colours={i: 'white', rand_idx: 'white'}, shuffling=True)
			array[i], array[rand_idx] = array[rand_idx], array[i]

		draw_array(array)
		pg.time.delay(2000)
		func(array)
		draw_array(array, default_colour='green')  # Draw final sorted array in green
		pg.time.delay(2000)
