# Mergesort and Quicksort demo
# Author: Sam Barba
# Created 06/09/2021

import random
from quicksort import *
from mergesort import *
from time import time

SIZE = 10 ** 5

nums = [random.randrange(SIZE) for i in range(SIZE)]
numsCopy = nums[:]

print("Sorting with mergesort...")
start = time()
mergesort(nums)
end = time()
timeTaken = round((end - start) * 1000)
print("Sorted in {} ms".format(timeTaken))

print("\nSorting with quicksort...")
start = time()
quicksort(numsCopy)
end = time()
timeTaken = round((end - start) * 1000)
print("Sorted in {} ms".format(timeTaken))
