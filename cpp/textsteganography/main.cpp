/*
LSB text steganography

Author: Sam Barba
Created 01/10/2021
*/

#include <bits/stdc++.h>
#include <bitset>
#include <iostream>
#include <string>

using std::bitset;
using std::cin;
using std::cout;
using std::setw;
using std::string;
using std::to_string;

const string ALPHANUMERIC_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

int setBit(const int n, const int idx, const int b) {
	// Set (idx)th bit of int 'n' to 'b'

	int mask = 1 << idx;
	return (n & ~mask) | (b << idx);
}

string decodeHiddenMsg(const string stegMsg) {
	uint8_t leastSignificantBits[stegMsg.length()];
	for (int i = 0; i < stegMsg.length(); i++)
		leastSignificantBits[i] = int(stegMsg[i]) & 1;

	// Read LSB array in chunks of size 8
	string decodedMsg = "";
	for (int i = 0; i < sizeof(leastSignificantBits); i += 8) {
		int ascii = 0;
		for (int j = i; j < i + 8; j++)
			ascii = (ascii << 1) | leastSignificantBits[j];
		decodedMsg += char(ascii);
	}

	return decodedMsg;
}

void hideMsg(const string msg) {
	// 1. Convert msg to binary

	string binaryMsg = "";
	for (char c : msg)
		binaryMsg += bitset<8>(c).to_string();

	cout << setw(35) << "In binary: ";
	cout << binaryMsg << '\n';

	// 2. Generate random container text

	string containerTxt = "";
	for (int i = 0; i < binaryMsg.length(); i++)
		containerTxt += ALPHANUMERIC_CHARS[rand() % (ALPHANUMERIC_CHARS.length() - 1)];

	cout << setw(35) << "Container text: ";
	cout << containerTxt << '\n';

	// 3. Hide message

	string hiddenMsg = "";
	for (int i = 0; i < containerTxt.length(); i++) {
		int ascii = int(containerTxt[i]);
		int bit = int(binaryMsg[i] - '0');
		int asciiNew = setBit(ascii, 0, bit);
		hiddenMsg += char(asciiNew);
	}

	cout << setw(35) << "Message hidden in container text: ";
	cout << hiddenMsg << '\n';

	// 4. Decode it back

	cout << setw(35) << "Decoded steganographic message: ";
	cout << decodeHiddenMsg(hiddenMsg) << "\n\n";
}

int main() {
	string msg;

	while (true) {
		cout << "Input message to hide (or X to exit)\n>>> ";
		getline(cin, msg);

		if (msg.length() == 1 && toupper(msg[0]) == 'X') break;

		hideMsg(msg);
	}
}
