/*
RSA demo

Author: Sam Barba
Created 14/06/2025
*/

#include <iostream>
#include <random>
#include <tuple>
#include <vector>

using std::cout;
using std::pair;
using std::string;
using std::vector;

std::random_device rd;
std::mt19937 gen(rd());

typedef pair<int, int> key;


int gcd(int a, int b) {
	// Greatest common divisor of a and b

	while (b) {
		int temp = b;
		b = a % b;
		a = temp;
	}

	return a;
}


int lcm(const int a, const int b) {
	// Lowest common multiple of a and b

	return a * b / gcd(a, b);
}


bool isCoprime(const int a, const int b) {
	return gcd(a, b) == 1;
}


std::tuple<int, int, int> extended_gcd(int a, int b) {
	// Returns a tuple (g, x, y) such that ax + by = g = gcd(a, b)

    if (a == 0)
        return std::make_tuple(b, 0, 1);
    int g, x, y;
    std::tie(g, y, x) = extended_gcd(b % a, a);

    return std::make_tuple(g, x - (b / a) * y, y);
}


int modInv(const int a, const int m) {
	// Modular multiplicative inverse

	int g, x, y;
	std::tie(g, x, y) = extended_gcd(a, m);
	if (g != 1)
		throw std::runtime_error("Modular inverse doesn't exist");
	return (x % m + m) % m;
}


pair<key, key> computePublicPrivateKeys(const int p, const int q) {
	// Compute public and private RSA keys given 2 (large) primes p, q

	int n = p * q;
	cout << "\np, q: " << p << ", " << q;
	cout << "\nn = pq: " << n;

	// Carmichael's totient function of n: lambda(n) = LCM(p - 1, q - 1)
	int lambda_n = lcm(p - 1, q - 1);
	cout << "\nlambda(n): " << lambda_n;

	// Choose any number 1 < e < lambda_n such that e is coprime to lambda_n
	vector<int> possible_e_vals;
	for (int e = 2; e < lambda_n; e++)
		if (isCoprime(e, lambda_n))
			possible_e_vals.push_back(e);
	cout << '\n' << possible_e_vals.size() << " possible values for e";
	std::uniform_int_distribution<int> dist(0, possible_e_vals.size() - 1);
	int e = possible_e_vals[dist(gen)];
	cout << "\nChosen e: " << e;

	int d = modInv(e, lambda_n);
	cout << "\nd: " << d;

	key publicKey(n, e);
	key privateKey(n, d);
	cout << "\nPublic key (n, e): (" << publicKey.first << ", " << publicKey.second << ')';
	cout << "\nPrivate key (n, d): (" << privateKey.first << ", " << privateKey.second << ')';

	return pair<key, key>(publicKey, privateKey);
}


int modExp(int base, int exp, int mod) {
	// Modular exponentiation

	int result = 1;
	base %= mod;
	while (exp > 0) {
		if (exp % 2)
			result = (result * base) % mod;
		base = (base * base) % mod;
		exp /= 2;
	}
	return result;
}


int encrypt(const int m, const key& publicKey) {
	// Encrypt a numeric plaintext message m with a public key (n, e)

	int n = publicKey.first;
	int e = publicKey.second;
	return modExp(m, e, n);
}


int decrypt(const int c, const key& privateKey) {
	// Decrypt a numeric ciphertext message c with a private key (n, d)

	int n = privateKey.first;
	int d = privateKey.second;
	return modExp(c, d, n);
}


int main() {
	int p = 53;
	int q = 61;
	pair<key, key>(keys) = computePublicPrivateKeys(p, q);
	key publicKey = keys.first;
	key privateKey = keys.second;

	string msg;

	while (true) {
		cout << "\n\nInput message (or X to exit)\n>>> ";
		getline(std::cin, msg);
		if (msg.length() == 1 && toupper(msg[0]) == 'X')
			break;

		vector<int> ascii(msg.length());
		vector<int> asciiEnc(msg.length());
		vector<int> asciiDec(msg.length());

		for (int i = 0; i < msg.length(); i++) {
			ascii[i] = int(msg[i]);
			asciiEnc[i] = encrypt(ascii[i], publicKey);
			asciiDec[i] = decrypt(asciiEnc[i], privateKey);
		}

		cout << "Message ASCII:";
		for (int i : ascii)
			cout << ' ' << i;

		cout << "\nEncrypted message ASCII:";
		for (int i : asciiEnc)
			cout << ' ' << i;

		cout << "\nDecrypted message ASCII:";
		for (int i : asciiDec)
			cout << ' ' << i;

		cout << "\nConverted back to text: ";
		for (int i : asciiDec)
			cout << char(i);
	}

	return 0;
}
