/*
Roman numeral converter

Author: Sam Barba
Created 06/09/2021
*/

#include <algorithm>
#include <iostream>
#include <vector>

using std::string;


const std::vector<std::pair<string, int>> NUMERAL_VALS = {
	{"M", 1000}, {"CM", 900}, {"D", 500}, {"CD", 400}, {"C", 100}, {"XC", 90},
	{"L", 50}, {"XL", 40}, {"X", 10}, {"IX", 9}, {"V", 5}, {"IV", 4}, {"I", 1}
};


bool isStringNumeric(const string s) {
	for (char c : s)
		if (c < '0' || c > '9') return false;

	return true;
}


string intToNumerals(int n) {
	if (n <= 0) return std::to_string(n);

	string numerals = "";
	for (const auto& item : NUMERAL_VALS) {
		string k = item.first;
		int v = item.second;
		while (n >= v) {
			numerals += k;
			n -= v;
		}
		if (!n) break;
	}

	return numerals;
}


int getValue(const string numeral) {
	for (const auto& item : NUMERAL_VALS)
		if (item.first == numeral)
			return item.second;
	return -1;
}


int numeralsToInt(const string numerals) {
	int n = 0;

	for (int i = 0; i < numerals.length(); i++) {
		string charToString = string(1, numerals[i]);
		int val = getValue(charToString);

		if (i + 1 < numerals.length() && getValue(string(1, numerals[i + 1])) > val)
			n -= val;
		else
			n += val;
	}

	return n;
}


string convert(const string input) {
	if (isStringNumeric(input))
		return intToNumerals(stoi(input));
	return std::to_string(numeralsToInt(input));
}


int main() {
	string input;

	while (true) {
		std::cout << "Input a number or numeral (or Q to quit)\n>>> ";
		std::cin >> input;
		transform(input.begin(), input.end(), input.begin(), ::toupper);

		if (input[0] == 'Q') break;
		std::cout << "Result: " << convert(input) << "\n\n";
	}

	return 0;
}
