/*
LSB text steganography

Author: Sam Barba
Created 01/10/2021
*/

#include <bitset>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>

using std::cout;
using std::setw;
using std::string;


const string ALPHANUMERIC_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dist(0, ALPHANUMERIC_CHARS.length() - 1);


int set_bit(const int n, const int idx, const int b) {
	// Set (idx)th bit of int 'n' to 'b'

	int mask = 1 << idx;
	return (n & ~mask) | (b << idx);
}


string decode_hidden_msg(const string& steg_msg) {
	uint8_t least_significant_bits[steg_msg.length()];
	for (int i = 0; i < steg_msg.length(); i++)
		least_significant_bits[i] = int(steg_msg[i]) & 1;

	// Read LSB array in chunks of size 8
	string decoded_msg = "";
	for (int i = 0; i < sizeof(least_significant_bits); i += 8) {
		int ascii = 0;
		for (int j = i; j < i + 8; j++)
			ascii = (ascii << 1) | least_significant_bits[j];
		decoded_msg += char(ascii);
	}

	return decoded_msg;
}


void hide_msg(const string& msg) {
	// 1. Convert msg to binary

	string binary_msg = "";
	for (char c : msg)
		binary_msg += std::bitset<8>(c).to_string();

	cout << setw(35) << "In binary: ";
	cout << binary_msg << '\n';

	// 2. Generate random container text

	string container_txt = "";
	for (int i = 0; i < binary_msg.length(); i++)
		container_txt += ALPHANUMERIC_CHARS[dist(gen)];

	cout << setw(35) << "Container text: ";
	cout << container_txt << '\n';

	// 3. Hide message

	string hidden_msg = "";
	for (int i = 0; i < container_txt.length(); i++) {
		int ascii = int(container_txt[i]);
		int bit = int(binary_msg[i] - '0');
		int ascii_new = set_bit(ascii, 0, bit);
		hidden_msg += char(ascii_new);
	}

	cout << setw(35) << "Message hidden in container text: ";
	cout << hidden_msg << '\n';

	// 4. Decode it back

	cout << setw(35) << "Decoded steganographic message: ";
	cout << decode_hidden_msg(hidden_msg) << "\n\n";
}


int main() {
	string msg;

	while (true) {
		cout << "Input message to hide (or X to exit)\n>>> ";
		getline(std::cin, msg);
		if (msg.length() == 1 && toupper(msg[0]) == 'X')
			break;
		hide_msg(msg);
	}

	return 0;
}
