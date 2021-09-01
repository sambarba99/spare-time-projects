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
