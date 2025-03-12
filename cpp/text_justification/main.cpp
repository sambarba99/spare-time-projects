/*
Program for solving 'Text Justification' LeetCode problem: https://leetcode.com/problems/text-justification/

Author: Sam Barba
Created 03/09/2022
*/

#include <cstring>
#include <iostream>
#include <vector>

using std::cout;
using std::string;
using std::vector;


vector<string> tokenise(const string text) {
	char charArray[text.length()];
	strcpy(charArray, text.c_str());
	char *ptr = strtok(charArray, " ");
	vector<string> words;

	while (ptr != NULL) {
		words.push_back(ptr);
		ptr = strtok(NULL, " ");
	}

	return words;
}


void justify(const vector<string>& words, const int maxWidth) {
	int nWords = words.size();
	int startIdx = 0;
	string line;
	vector<string> res;

	while (true) {
		int counter = startIdx;
		if (counter >= nWords) break;

		int lineLen = 0;
		// Find start/end word indices for one line
		while (counter < nWords) {
			lineLen += words[counter].length();
			if (counter != startIdx) lineLen++;
			if (lineLen > maxWidth) break;
			counter++;
		}

		// Justify one line
		if (counter != nWords) {
			int endIdx = counter - 1;
			if (startIdx == endIdx) {  // One word in line
				line = words[startIdx];
				for (int i = 0; i < maxWidth - words[startIdx].length(); i++)
					line += " ";
			} else {  // Many words in line
				lineLen -= words[counter].length() + 1;
				int wordNum = endIdx - startIdx + 1;
				int extraSpaces = maxWidth - (lineLen - (wordNum - 1));
				int basicPadSpaces = extraSpaces / (wordNum - 1);
				int additionPadSpaces = extraSpaces % (wordNum - 1);
				line = "";
				for (int i = startIdx; i < counter - 1; i++) {
					line += words[i];
					for (int j = 0; j < basicPadSpaces; j++)
						line += " ";
					if (i - startIdx < additionPadSpaces)
						line += " ";
				}
				line += words[counter - 1];
			}
		} else {  // Last line
			line = "";
			for (int i = startIdx; i < nWords; i++)
				line += words[i] + " ";
		}

		res.push_back(line);
		startIdx = counter;
	}

	for (const string line : res)
		cout << line << '\n';
}


int main() {
	string text;
	int maxWidth;

	cout << "Input text to justify\n>>> ";
	getline(std::cin, text);

	cout << "Input max. width\n>>> ";
	std::cin >> maxWidth;

	vector<string> words = tokenise(text);
	cout << "\nJustified:\n";
	justify(words, maxWidth);

	return 0;
}
