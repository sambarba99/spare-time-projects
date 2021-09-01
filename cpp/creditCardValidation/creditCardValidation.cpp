/*
Credit card validation demo using Luhn algorithm

Author: Sam Barba
Created 10/08/2022
*/

#include <iostream>
#include <string>

using namespace std;

bool isStringNumeric(const string s) {
	for (int i = 0; i < s.length(); i++) {
		if (s[i] < '0' || s[i] > '9') return false;
	}
	return true;
}

bool isCardNumberValid(const string cardNumber) {
	// 1. Double every even-placed digit. If the result is a 2-digit number, add both
	// digits to obtain a single digit number. Finally, sum all these results.

	int len = cardNumber.length();
	int sum = 0;
	for (int i = 0; i < len; i += 2) {
		int dbl = (cardNumber[i] - 48) * 2;
		if (dbl > 9) dbl -= 9;  // Same as adding both digits together
		sum += dbl;
	}

	// 2. Add every odd-placed digit to 'sum'

	for (int i = 1; i < len; i += 2) {
		sum += cardNumber[i] - 48;
	}

	// 3. If 'sum' is a multiple of 10, cardNumber is valid. Otherwise, not.

	return sum % 10 == 0;
}

int main() {
	string cardNumber;

	while (true) {
		cout << "Enter a card number to validate (or 'x' to exit): ";
		cin >> cardNumber;

		if (cardNumber == "x" || cardNumber == "X") break;
		else if (!isStringNumeric(cardNumber)) {
			cout << "Input not numeric!\n";
			continue;
		}

		bool isValid = isCardNumberValid(cardNumber);

		cout << (isValid ? "Card number valid" : "Card number invalid") << "\n";
	}
}
