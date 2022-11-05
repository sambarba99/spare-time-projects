/*
Number to words converter

Author: Sam Barba
Created 06/09/2022
*/

#include <cmath>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <vector>

using std::cin;
using std::cout;
using std::map;
using std::pair;
using std::regex;
using std::stold;
using std::string;
using std::vector;

map<int, string> SMALL = {
	{0, ""}, {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}, {5, "five"}, {6, "six"}, {7, "seven"}, {8, "eight"},
	{9, "nine"}, {10, "ten"}, {11, "eleven"}, {12, "twelve"}, {13, "thirteen"}, {14, "fourteen"}, {15, "fifteen"},
	{16, "sixteen"}, {17, "seventeen"}, {18, "eighteen"}, {19, "nineteen"}
};

map<int, string> TENS = {
	{2, "twenty"}, {3, "thirty"}, {4, "forty"}, {5, "fifty"}, {6, "sixty"}, {7, "seventy"}, {8, "eighty"}, {9, "ninety"}
};

const vector<pair<int, string>> BIG = {
	{1, "thousand"}, {2, "million"}, {3, "billion"}, {4, "trillion"}, {5, "quadrillion"}, {6, "quintillion"},
	{7, "sextillion"}, {8, "septillion"}, {9, "octillion"}, {10, "nonillion"}, {11, "decillion"}
};

string join(const vector<string> strings) {
	string result = "";
	for (vector<string>::const_iterator it = strings.begin(); it != strings.end(); it++)
		if (*it != "") result += *it + " ";
	result = regex_replace(result, regex(R"(\s-\s)"), "-");  // twenty - one -> twenty-one
	result = regex_replace(result, regex(R"(\s\s)"), " ");  // Remove any duplicate spaces
	return result;
}

string say_num_pos(const long double n) {
	if (n < 20) return SMALL[n];
	else if (n < 100) {
		// These nums should be hyphenated (unless ending with 0), e.g. 21 -> 'twenty-one'
		int lastDigit = fmod(n, 10);
		if (lastDigit == 0) return TENS[n / 10];
		return join({TENS[n / 10], "-", SMALL[lastDigit]});
	} else if (n < 1000) {
		return join({
			say_num_pos(n / 100),
			"hundred",
			say_num_pos(fmod(n, 100))
		});
	} else {
		int illionsNum;
		string illionsName;
		for (auto it : BIG) {
			illionsNum = it.first;
			if (n < pow(1000, illionsNum + 1)) {
				illionsName = it.second;
				break;
			}
		}
		return join({
			say_num_pos(n / pow(1000, illionsNum)),
			illionsName,
			say_num_pos(fmod(n, pow(1000, illionsNum)))
		});
	}
}

string convert(const long double n) {
	if (n < 0) return join({"minus", say_num_pos(-n)});
	if (n == 0) return "zero";
	return say_num_pos(n);
}

int main() {
	string n;

	while (true) {
		cout << "Input a number (or X to exit)\n>>> ";
		cin >> n;

		if (toupper(n[0]) == 'X') break;
		else cout << convert(stold(n)) << "\n\n";
	}
}
