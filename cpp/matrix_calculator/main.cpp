/*
Matrix calculator

Author: Sam Barba
Created 03/09/2021
*/

#include <iostream>
#include <iterator>
#include <sstream>

#include "matrix.h"

using std::cin;
using std::cout;
using std::istream_iterator;
using std::string;


vector<int> getDimsFromStr(const string dimStr) {
	std::stringstream ss(dimStr);
	auto start = istream_iterator<int>{ss};
	auto end = istream_iterator<int>{};
	vector<int> dims(start, end); 
	return dims;
}


vector<long double> vectorFromLine(const string line) {
	// Convert line to 1D vector of long doubles
	std::stringstream ss(line);
	auto start = istream_iterator<long double>{ss};
	auto end = istream_iterator<long double>{};
	vector<long double> v(start, end);

	return v;
}


vector<vector<long double>> gridFromLine(const string line, const int rows, const int cols) {
	vector<long double> grid1d = vectorFromLine(line);
	vector<vector<long double>> grid2d(rows, vector<long double>(cols));
	for (int i = 0; i < rows * cols; i++) {
		int row = i / cols, col = i % cols;
		grid2d[row][col] = grid1d[i];
	}

	return grid2d;
}


string getExample(const int length) {
	string e = "";
	for (int i = 1; i < length + 1; i++)
		e += std::to_string(i) + ' ';
	return e;
}


int main() {
	char choice;
	string dimStrMatA, dimStrMatB, matAstr, matBstr;

	while (true) {
		cout << "Enter: A to add/subtract matrices,\n";
		cout << " M to multiply matrices,\n";
		cout << " D to divide matrices,\n";
		cout << " P to do matrix to a power,\n";
		cout << " R to convert a matrix to reduced row echelon form,\n";
		cout << " I to find the comatrix, inverse & determinant of a matrix,\n";
		cout << " G to do geometric transformations,\n";
		cout << " or X to exit:\n>>> ";
		cin >> choice;
		choice = toupper(choice);
		cin.clear();  // Reset to use getline afterwards
		cin.sync();
		cout << '\n';

		if (choice == 'A') {
			cout << "Input the number of rows & columns for the matrices e.g. 2 3\n>>> ";
			getline(cin, dimStrMatA);
			vector<int> dims = getDimsFromStr(dimStrMatA);
			int rows = dims[0], cols = dims[1];

			cout << "Input matrix A (" << (rows * cols) << " entries) e.g. " << getExample(rows * cols) << "\n>>> ";
			getline(cin, matAstr);
			cout << "Input matrix B (" << (rows * cols) << " entries) e.g. " << getExample(rows * cols) << "\n>>> ";
			getline(cin, matBstr);

			Matrix matA(gridFromLine(matAstr, rows, cols));
			Matrix matB(gridFromLine(matBstr, rows, cols));

			cout << "\nA + B =\n" << matA.addSubtract(matB, true).toString();
			cout << "\nA - B =\n" << matA.addSubtract(matB, false).toString() << '\n';

		} else if (choice == 'M') {
			cout << "Input the number of rows & columns for matrix A e.g. 2 3\n>>> ";
			getline(cin, dimStrMatA);
			vector<int> dimsA = getDimsFromStr(dimStrMatA);
			int rowsA = dimsA[0], colsA = dimsA[1];

			cout << "Input the number of columns for matrix B (rows = " << colsA << ")\n>>> ";
			getline(cin, dimStrMatB);
			vector<int> dimsB = getDimsFromStr(dimStrMatB);
			int rowsB = colsA, colsB = dimsB[0];

			cout << "Input matrix A (" << (rowsA * colsA) << " entries) e.g. " << getExample(rowsA * colsA) << "\n>>> ";
			getline(cin, matAstr);
			cout << "Input matrix B (" << (rowsB * colsB) << " entries) e.g. " << getExample(rowsB * colsB) << "\n>>> ";
			getline(cin, matBstr);

			Matrix matA(gridFromLine(matAstr, rowsA, colsA));
			Matrix matB(gridFromLine(matBstr, rowsB, colsB));

			cout << "\nA x B =\n" << matA.mult(matB).toString() << '\n';

		} else if (choice == 'D') {
			cout << "Input the number of rows & columns for matrix A e.g. 2 3\n>>> ";
			getline(cin, dimStrMatA);
			vector<int> dimsA = getDimsFromStr(dimStrMatA);
			int rowsA = dimsA[0], colsA = dimsA[1];
			int rowsB = colsA, colsB = colsA;
			cout << "B is " << rowsB << 'x' << colsB << "\n\n";

			cout << "Input matrix A (" << (rowsA * colsA) << " entries) e.g. " << getExample(rowsA * colsA) << "\n>>> ";
			getline(cin, matAstr);
			cout << "Input matrix B (" << (rowsB * colsB) << " entries) e.g. " << getExample(rowsB * colsB) << "\n>>> ";
			getline(cin, matBstr);

			Matrix matA(gridFromLine(matAstr, rowsA, colsA));
			Matrix matB(gridFromLine(matBstr, rowsB, colsB));

			if (matB.determinant() == 0)
				cout << "\nCan't divide (determinant of B = 0)\n";
			else
				cout << "\nA / B =\n" << matA.mult(matB.inverse()).toString() << '\n';

		} else if (choice == 'P') {
			cout << "Input the size of the square matrix e.g. 2\n>>> ";
			getline(cin, dimStrMatA);
			int size = stoi(dimStrMatA);
			cout << "Input the power\n>>> ";
			string pStr;
			getline(cin, pStr);
			int p = stoi(pStr);

			cout << "Input matrix M (" << (size * size) << " entries) e.g. " << getExample(size * size) << "\n>>> ";
			getline(cin, matAstr);

			Matrix mat(gridFromLine(matAstr, size, size));

			cout << "\nM^" << p << "=\n" << mat.power(p).toString() << '\n';

		} else if (choice == 'R') {
			cout << "Input the number of rows & columns for the matrix e.g. 2 3\n>>> ";
			getline(cin, dimStrMatA);
			vector<int> dims = getDimsFromStr(dimStrMatA);
			int rows = dims[0], cols = dims[1];

			cout << "Input matrix M (" << (rows * cols) << " entries) e.g. " << getExample(rows * cols) << "\n>>> ";
			getline(cin, matAstr);

			Matrix mat(gridFromLine(matAstr, rows, cols));

			cout << "\nRREF(M) =\n" << mat.rref().toString() << '\n';

		} else if (choice == 'I') {
			cout << "Input the size of the square matrix e.g. 2\n>>> ";
			getline(cin, dimStrMatA);
			int size = stoi(dimStrMatA);

			cout << "Input matrix M (" << (size * size) << " entries) e.g. " << getExample(size * size) << "\n>>> ";
			getline(cin, matAstr);

			Matrix mat(gridFromLine(matAstr, size, size));

			cout << "\nComatrix =\n" << mat.comatrix().toString() << '\n';

			long double det = mat.determinant();
			if (det == 0)
				cout << "No inverse (determinant = 0)\n";
			else
				cout << "Inverse:\n" << mat.inverse().toString() << "\nDeterminant: " << det << "\n\n";

		} else if (choice == 'G') {
			cout << "Enter 1 for translation, 2 for enlargement, 3 for reflection, or 4 for rotation\n>>> ";
			string geomChoice;
			getline(cin, geomChoice);

			cout << "How many vertices?\n>>> ";
			string nVertsStr;
			getline(cin, nVertsStr);

			vector<vector<long double>> coords;
			string str;
			for (int i = 0; i < stoi(nVertsStr); i++) {
				cout << "Input x,y coords " << (i + 1) << '/' << nVertsStr << " e.g. " << getExample(2) << "\n>>> ";
				getline(cin, str);
				coords.push_back(vectorFromLine(str));
			}

			Matrix mat(coords);

			if (geomChoice == "1") {
				cout << "Input the change in x and change in y e.g. " << getExample(2) << "\n>>> ";
				getline(cin, str);
				vector<long double> v = vectorFromLine(str);
				long double dx = v[0], dy = v[1];
				cout << "\nResultant coords:\n" << mat.translate(dx, dy).toString() << '\n';
			} else if (geomChoice == "2") {
				cout << "Input the enlargement factor and x, y e.g. 2 3 4\n>>> ";
				getline(cin, str);
				vector<long double> v = vectorFromLine(str);
				long double k = v[0], x = v[1], y = v[2];
				cout << "\nResultant coords:\n" << mat.enlarge(k, x, y).toString() << '\n';
			} else if (geomChoice == "3") {
				cout << "Input m, c for reflection line y = mx + c e.g. 2 3\n>>> ";
				getline(cin, str);
				vector<long double> v = vectorFromLine(str);
				long double m = v[0], c = v[1];
				cout << "\nResultant coords:\n" << mat.reflect(m, c).toString() << '\n';
			} else if (geomChoice == "4") {
				cout << "Input the clockwise rotation angle (deg) and x, y e.g. 90 0 2\n>>> ";
				getline(cin, str);
				vector<long double> v = vectorFromLine(str);
				long double a = v[0], x = v[1], y = v[2];
				cout << "\nResultant coords:\n" << mat.rotate(a, x, y).toString() << '\n';
			}

		} else if (choice == 'X') {
			break;
		}
	}

	return 0;
}
