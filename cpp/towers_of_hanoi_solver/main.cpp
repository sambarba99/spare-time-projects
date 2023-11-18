/*
Towers of Hanoi solver

Author: Sam Barba
Created 20/09/2021
*/

#include <iostream>

using std::cin;
using std::cout;

int stepNum = 0;

void solve(const int n, const int t1 = 1, const int t2 = 2, const int t3 = 3) {
	if (!n) return;

	solve(n - 1, t1, t3, t2);
	stepNum++;
	cout << "Step " << stepNum << ": move disc from " << t1 << " to " << t3 << '\n';
	solve(n - 1, t2, t1, t3);
}

int main() {
	int nDiscs;
	cout << "Input no. discs\n>>> ";
	cin >> nDiscs;
	solve(nDiscs);
}
