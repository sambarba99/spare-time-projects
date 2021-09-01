/*
C++ basics demo

Author: Sam Barba
Created 09/08/2022
*/

#include <iostream>
#include <string>

using namespace std;

enum MyEnum {
	OPT1,
	OPT2,
	OPT3
};

struct MyStruct {
	string field1;  // Fields a.k.a. members
	MyEnum field2;
	bool field3;
};

int main() {
	// For loop printing
	int lim = 2e4;
	for (int i = 0; i <= lim; i++) {
		cout << '\n' << i;
	}

	// IO basics
	float x, y;
	cout << "\n\nEnter 2 numbers: ";
	cin >> x >> y;
	float sum = x + y;
	cout << "Their sum is " << sum;

	// Pointers
	char c = 'S';
	char* address = &c;  // Address of c
	cout << "\n\n" << *address;

	// Enums and structs
	MyStruct structInstance = {"field1 value", OPT1, true};
	cout << "\n\n" << structInstance.field1;
}
