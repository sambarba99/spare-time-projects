/*
Tower of Hanoi solver

Author: Sam Barba
Created 20/09/2021
*/

#include <iostream>


int step_num = 0;


void solve(const int n, const int t1 = 1, const int t2 = 2, const int t3 = 3) {
	if (!n) return;

	solve(n - 1, t1, t3, t2);
	step_num++;
	std::cout << "Step " << step_num << ": move disc from " << t1 << " to " << t3 << '\n';
	solve(n - 1, t2, t1, t3);
}


int main() {
	int num_discs;
	std::cout << "Input no. discs\n>>> ";
	std::cin >> num_discs;
	solve(num_discs);

	return 0;
}
