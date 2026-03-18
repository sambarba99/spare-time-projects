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


// The prime numbers we'll use in this demo
const int P = 53;
const int Q = 61;

using Key = pair<int, int>;

std::random_device rd;
std::mt19937 gen(rd());


bool is_prime(const int n) {
	if (n < 2)
		return false;
	if (n <= 3)
		return true;
	if (n % 2 == 0 || n % 3 == 0)
		return false;
	int limit = static_cast<int>(std::sqrt(n));
	for (int i = 5; i <= limit; i += 6)
		if (n % i == 0 || n % (i + 2) == 0)
			return false;
	return true;
}


int gcd(int a, int b) {
	// Greatest common divisor of a and b

	int temp;
	while (b) {
		temp = b;
		b = a % b;
		a = temp;
	}

	return a;
}


int lcm(const int a, const int b) {
	// Lowest common multiple of a and b

	return a * b / gcd(a, b);
}


bool is_coprime(const int a, const int b) {
	return gcd(a, b) == 1;
}


std::tuple<int, int, int> extended_gcd(const int a, const int b) {
	// Returns a tuple (r, x, y) such that ax + by = r = gcd(a, b)

	int old_r = a, r = b;
	int old_x = 1, x = 0;
	int old_y = 0, y = 1;
	int q, temp_r, temp_x, temp_y;

	while (r != 0) {
		q = old_r / r;

		temp_r = r;
		r = old_r - q * r;
		old_r = temp_r;

		temp_x = x;
		x = old_x - q * x;
		old_x = temp_x;

		temp_y = y;
		y = old_y - q * y;
		old_y = temp_y;
	}

	return std::make_tuple(old_r, old_x, old_y);
}


int modular_multiplicative_inverse(const int a, const int m) {
	// Returns an integer x such that ax = 1 (mod m)

	int g, x, y;
	std::tie(g, x, y) = extended_gcd(a, m);
	if (g != 1)
		throw std::runtime_error("Modular inverse doesn't exist");
	return (x % m + m) % m;
}


pair<Key, Key> compute_public_private_keys(const int p, const int q) {
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
		if (is_coprime(e, lambda_n))
			possible_e_vals.emplace_back(e);
	cout << '\n' << possible_e_vals.size() << " possible values for e";
	std::uniform_int_distribution<int> dist(0, possible_e_vals.size() - 1);
	int e = possible_e_vals[dist(gen)];
	cout << "\nChosen e: " << e;

	int d = modular_multiplicative_inverse(e, lambda_n);
	cout << "\nd: " << d;

	Key public_key(n, e);
	Key private_key(n, d);
	cout << "\nPublic key (n, e): (" << public_key.first << ", " << public_key.second << ')';
	cout << "\nPrivate key (n, d): (" << private_key.first << ", " << private_key.second << ')';

	return {public_key, private_key};
}


int modular_exp(int base, int exp, int mod) {
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


int encrypt(const int m, const Key& public_key) {
	// Encrypt a numeric plaintext message m with a public key (n, e)

	int n = public_key.first;
	int e = public_key.second;
	return modular_exp(m, e, n);
}


int decrypt(const int c, const Key& private_key) {
	// Decrypt a numeric ciphertext message c with a private key (n, d)

	int n = private_key.first;
	int d = private_key.second;
	return modular_exp(c, d, n);
}


int main() {
	if (!is_prime(P) || !is_prime(Q)) {
		cout << "P and Q must be prime";
		return 0;
	}

	pair<Key, Key>(keys) = compute_public_private_keys(P, Q);
	Key public_key = keys.first;
	Key private_key = keys.second;

	string msg;

	while (true) {
		cout << "\n\nInput message (or X to exit)\n>>> ";
		getline(std::cin, msg);
		if (msg.length() == 1 && toupper(msg[0]) == 'X')
			break;

		vector<int> ascii(msg.length());
		vector<int> ascii_enc(msg.length());
		vector<int> ascii_dec(msg.length());

		for (int i = 0; i < msg.length(); i++) {
			ascii[i] = int(msg[i]);
			ascii_enc[i] = encrypt(ascii[i], public_key);
			ascii_dec[i] = decrypt(ascii_enc[i], private_key);
		}

		cout << "Message ASCII:";
		for (int i : ascii)
			cout << ' ' << i;

		cout << "\nEncrypted message ASCII:";
		for (int i : ascii_enc)
			cout << ' ' << i;

		cout << "\nDecrypted message ASCII:";
		for (int i : ascii_dec)
			cout << ' ' << i;

		cout << "\nConverted back to text: ";
		for (int i : ascii_dec)
			cout << char(i);
	}

	return 0;
}
