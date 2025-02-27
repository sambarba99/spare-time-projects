/*
Pi Calculator

Author: Sam Barba
Created 04/03/2019
*/

#include <cmath>
#include <iostream>

using std::cout;


void calcPi(const int numDigits) {
	long double a = 0, b = 1, c = 1, d = 1, e = 3, n = 3;
	long double na, nn;
	int count = 0;

	cout << "First " << numDigits << " digits of pi: ";

	while (count < numDigits) {
		if (4 * b + a - c < n * c) {
			cout << n;
			count++;
			na = 10 * (a - n * c);
			n = floor(((10 * (3 * b + a)) / c) - 10 * n);
			b *= 10;
		} else {
			na = (2 * b + a) * e;
			nn = floor((b * (7 * d) + 2 + (a * e)) / (c * e));
			b *= d;
			c *= e;
			e += 2;
			d++;
			n = nn;
		}
		a = na;
	}
}


int main() {
	int numDigits = 100;
	calcPi(numDigits);
}
