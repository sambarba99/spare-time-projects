/*
Program for solving 'Trapping Rain Water' LeetCode problem: https://leetcode.com/problems/trapping-rain-water/
in O(n) time

Author: Sam Barba
Created 03/09/2022
*/

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using std::cin;
using std::cout;
using std::max;
using std::string;
using std::vector;
using std::getline;

int trap(const vector<int> heights) {
	if (heights.size() == 0) return 0;

	int l = 0, r = heights.size() - 1;
	int leftMax = heights[l], rightMax = heights[r];
	int res = 0;

	while (l < r) {
		if (leftMax < rightMax) {
			l++;
			leftMax = max(leftMax, heights[l]);
			res += leftMax - heights[l];
		} else {
			r--;
			rightMax = max(rightMax, heights[r]);
			res += rightMax - heights[r];
		}
	}

	return res;
}

int main() {
	string input;
	vector<int> heights;

	cout << "Input wall heights e.g. 5,4,2,5\n>>> ";
	getline(cin, input);
	for (char c : input)
		if (c != ',') heights.push_back(c - '0');

	int result = trap(heights);
	cout << "\nThose heights can trap " << result << " units of water";
}
