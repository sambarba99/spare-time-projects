/*
Number to words converter

Author: Sam Barba
Created 06/09/2022
*/

#include <cmath>
#include <iostream>
#include <regex>
#include <unordered_map>

using std::string;
using std::vector;


const std::unordered_map<int, string> SMALL = {
	{0, ""}, {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}, {5, "five"}, {6, "six"}, {7, "seven"}, {8, "eight"},
	{9, "nine"}, {10, "ten"}, {11, "eleven"}, {12, "twelve"}, {13, "thirteen"}, {14, "fourteen"}, {15, "fifteen"},
	{16, "sixteen"}, {17, "seventeen"}, {18, "eighteen"}, {19, "nineteen"}
};

const std::unordered_map<int, string> TENS = {
	{2, "twenty"}, {3, "thirty"}, {4, "forty"}, {5, "fifty"}, {6, "sixty"}, {7, "seventy"}, {8, "eighty"}, {9, "ninety"}
};

const vector<std::pair<int, string>> BIG = {
	{1, "thousand"}, {2, "million"}, {3, "billion"}, {4, "trillion"}, {5, "quadrillion"}, {6, "quintillion"},
	{7, "sextillion"}, {8, "septillion"}, {9, "octillion"}, {10, "nonillion"}, {11, "decillion"}
};


string join(const vector<string>& strings) {
	string result = "";
	for (vector<string>::const_iterator it = strings.begin(); it != strings.end(); it++)
		if (*it != "")
			result += *it + " ";
	result = regex_replace(result, std::regex(R"(\s-\s)"), "-");  // twenty - one -> twenty-one
	result = regex_replace(result, std::regex(R"(\s\s)"), " ");  // Remove any duplicate spaces
	return result;
}


string say_num_pos(const long double n) {
	if (n < 20)
		return SMALL.at(n);
	else if (n < 100) {
		// These nums should be hyphenated (unless ending with 0), e.g. 21 -> 'twenty-one'
		int last_digit = fmod(n, 10);
		if (last_digit == 0)
			return TENS.at(n / 10);
		return join({TENS.at(n / 10), "-", SMALL.at(last_digit)});
	} else if (n < 1000) {
		return join({
			say_num_pos(n / 100),
			"hundred",
			say_num_pos(fmod(n, 100))
		});
	} else {
		int illions_num;
		string illions_name;
		for (const auto& [illions_num_i, ilions_name_i] : BIG) {
			illions_num = illions_num_i;
			if (n < pow(1000, illions_num + 1)) {
				illions_name = ilions_name_i;
				break;
			}
		}
		return join({
			say_num_pos(n / pow(1000, illions_num)),
			illions_name,
			say_num_pos(fmod(n, pow(1000, illions_num)))
		});
	}
}


string convert(const long double n) {
	if (n < 0)
		return join({"minus", say_num_pos(-n)});
	if (n == 0)
		return "zero";
	return say_num_pos(n);
}


int main() {
	string n;

	while (true) {
		std::cout << "Input a number (or X to exit)\n>>> ";
		std::cin >> n;

		if (toupper(n[0]) == 'X')
			break;
		else
			std::cout << convert(stold(n)) << "\n\n";
	}

	return 0;
}
