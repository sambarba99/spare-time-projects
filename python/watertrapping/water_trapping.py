"""
Program for solving 'Trapping Rain Water' LeetCode problem: https://leetcode.com/problems/trapping-rain-water/
in O(n) time

Author: Sam Barba
Created 03/09/2022
"""

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def trap(heights):
	if len(heights) == 0: return

	l, r = 0, len(heights) - 1
	left_max, right_max = heights[l], heights[r]
	res = 0
	water_heights = [0] * len(heights)

	while l < r:
		if left_max < right_max:
			l += 1
			left_max = max(left_max, heights[l])
			res += left_max - heights[l]
			water_heights[l] = left_max
		else:
			r -= 1
			right_max = max(right_max, heights[r])
			res += right_max - heights[r]
			water_heights[r] = right_max

	plt.bar(range(len(heights)), heights, width=1, color='black', zorder=2, label='Walls')
	plt.bar(range(len(heights)), water_heights, width=1, color='#0080ff', zorder=1, label='Water')
	plt.axis('scaled')
	plt.title(f'Heights: {heights}\ncan trap {res} units of water')
	plt.legend()
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	wall_heights = np.random.randint(0, 6, size=8)
	trap(wall_heights)
