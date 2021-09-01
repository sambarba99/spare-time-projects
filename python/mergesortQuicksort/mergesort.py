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
