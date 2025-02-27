/*
Program for solving 'Trapping Rain Water' LeetCode problem: https://leetcode.com/problems/trapping-rain-water/
in O(n) time

Author: Sam Barba
Created 03/09/2022
*/

#include <iostream>
#include <sstream>
#include <vector>


int trap(const std::vector<int>& heights) {
	int l = 0, r = heights.size() - 1;
	int leftMax = heights[l], rightMax = heights[r];
	int res = 0;

	while (l < r) {
		if (leftMax < rightMax) {
			l++;
			leftMax = std::max(leftMax, heights[l]);
			res += leftMax - heights[l];
		} else {
			r--;
			rightMax = std::max(rightMax, heights[r]);
			res += rightMax - heights[r];
		}
	}

	return res;
}


int main() {
	std::string input;
	std::vector<int> heights;

	std::cout << "Input wall heights e.g. 5,4,2,5\n>>> ";
	getline(std::cin, input);
	std::stringstream ss(input);
	std::string temp;
	while (getline(ss, temp, ','))
		heights.push_back(std::stoi(temp));

	int result = trap(heights);
	std::cout << "\nThose heights can trap " << result << " units of water";

	return 0;
}
