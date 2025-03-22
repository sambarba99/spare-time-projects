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


bool is_string_numeric(const string& s) {
	for (char c : s)
		if (!isdigit(c))
			return false;

	return true;
}


string int_to_numerals(int n) {
	if (n <= 0) return std::to_string(n);

	string numerals = "";
	for (const auto& [k, v] : NUMERAL_VALS) {
		while (n >= v) {
			numerals += k;
			n -= v;
		}
		if (!n) break;
	}

	return numerals;
}


int get_value(const string& numeral) {
	for (const auto& [k, v] : NUMERAL_VALS)
		if (k == numeral)
			return v;
	throw std::exception();
}


int numerals_to_int(const string& numerals) {
	int n = 0;

	for (int i = 0; i < numerals.length(); i++) {
		string char_to_string = string(1, numerals[i]);
		int val = get_value(char_to_string);

		if (i + 1 < numerals.length() && get_value(string(1, numerals[i + 1])) > val)
			n -= val;
		else
			n += val;
	}

	return n;
}


string convert(const string& input) {
	if (is_string_numeric(input))
		return int_to_numerals(stoi(input));
	return std::to_string(numerals_to_int(input));
}


int main() {
	string input;

	while (true) {
		std::cout << "Input a number or numeral (or Q to quit)\n>>> ";
		std::cin >> input;
		transform(input.begin(), input.end(), input.begin(), ::toupper);

		if (input[0] == 'Q')
			break;
		std::cout << "Result: " << convert(input) << "\n\n";
	}

	return 0;
}
