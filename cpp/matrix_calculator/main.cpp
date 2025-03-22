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


vector<int> get_dims_from_str(const string& dim_str) {
	std::stringstream ss(dim_str);
	auto start = istream_iterator<int>{ss};
	auto end = istream_iterator<int>{};
	vector<int> dims(start, end); 
	return dims;
}


vector<long double> vector_from_line(const string& line) {
	// Convert line to 1D vector of long doubles
	std::stringstream ss(line);
	auto start = istream_iterator<long double>{ss};
	auto end = istream_iterator<long double>{};
	vector<long double> v(start, end);

	return v;
}


vector<vector<long double>> grid_from_line(const string& line, const int rows, const int cols) {
	vector<long double> grid_1d = vector_from_line(line);
	vector<vector<long double>> grid_2d(rows, vector<long double>(cols));
	for (int i = 0; i < rows * cols; i++) {
		int row = i / cols, col = i % cols;
		grid_2d[row][col] = grid_1d[i];
	}

	return grid_2d;
}


string get_example(const int length) {
	string e = "";
	for (int i = 1; i < length + 1; i++)
		e += std::to_string(i) + ' ';
	return e;
}


int main() {
	char choice;
	string dim_str_mat_a, dim_str_mat_b, mat_a_str, mat_b_str;

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
			getline(cin, dim_str_mat_a);
			vector<int> dims = get_dims_from_str(dim_str_mat_a);
			int rows = dims[0], cols = dims[1];

			cout << "Input matrix A (" << (rows * cols) << " entries) e.g. " << get_example(rows * cols) << "\n>>> ";
			getline(cin, mat_a_str);
			cout << "Input matrix B (" << (rows * cols) << " entries) e.g. " << get_example(rows * cols) << "\n>>> ";
			getline(cin, mat_b_str);

			Matrix matA(grid_from_line(mat_a_str, rows, cols));
			Matrix matB(grid_from_line(mat_b_str, rows, cols));

			cout << "\nA + B =\n" << matA.add_subtract(matB, true).to_string();
			cout << "\nA - B =\n" << matA.add_subtract(matB, false).to_string() << '\n';

		} else if (choice == 'M') {
			cout << "Input the number of rows & columns for matrix A e.g. 2 3\n>>> ";
			getline(cin, dim_str_mat_a);
			vector<int> dims_a = get_dims_from_str(dim_str_mat_a);
			int rows_a = dims_a[0], cols_a = dims_a[1];

			cout << "Input the number of columns for matrix B (rows = " << cols_a << ")\n>>> ";
			getline(cin, dim_str_mat_b);
			vector<int> dims_b = get_dims_from_str(dim_str_mat_b);
			int rows_b = cols_a, cols_b = dims_b[0];

			cout << "Input matrix A (" << (rows_a * cols_a) << " entries) e.g. " << get_example(rows_a * cols_a) << "\n>>> ";
			getline(cin, mat_a_str);
			cout << "Input matrix B (" << (rows_b * cols_b) << " entries) e.g. " << get_example(rows_b * cols_b) << "\n>>> ";
			getline(cin, mat_b_str);

			Matrix matA(grid_from_line(mat_a_str, rows_a, cols_a));
			Matrix matB(grid_from_line(mat_b_str, rows_b, cols_b));

			cout << "\nA x B =\n" << matA.mult(matB).to_string() << '\n';

		} else if (choice == 'D') {
			cout << "Input the number of rows & columns for matrix A e.g. 2 3\n>>> ";
			getline(cin, dim_str_mat_a);
			vector<int> dims_a = get_dims_from_str(dim_str_mat_a);
			int rows_a = dims_a[0], cols_a = dims_a[1];
			int rows_b = cols_a, cols_b = cols_a;
			cout << "B is " << rows_b << 'x' << cols_b << "\n\n";

			cout << "Input matrix A (" << (rows_a * cols_a) << " entries) e.g. " << get_example(rows_a * cols_a) << "\n>>> ";
			getline(cin, mat_a_str);
			cout << "Input matrix B (" << (rows_b * cols_b) << " entries) e.g. " << get_example(rows_b * cols_b) << "\n>>> ";
			getline(cin, mat_b_str);

			Matrix matA(grid_from_line(mat_a_str, rows_a, cols_a));
			Matrix matB(grid_from_line(mat_b_str, rows_b, cols_b));

			if (matB.determinant() == 0)
				cout << "\nCan't divide (determinant of B = 0)\n";
			else
				cout << "\nA / B =\n" << matA.mult(matB.inverse()).to_string() << '\n';

		} else if (choice == 'P') {
			cout << "Input the size of the square matrix e.g. 2\n>>> ";
			getline(cin, dim_str_mat_a);
			int size = stoi(dim_str_mat_a);
			cout << "Input the power\n>>> ";
			string p_str;
			getline(cin, p_str);
			int p = stoi(p_str);

			cout << "Input matrix M (" << (size * size) << " entries) e.g. " << get_example(size * size) << "\n>>> ";
			getline(cin, mat_a_str);

			Matrix mat(grid_from_line(mat_a_str, size, size));

			cout << "\nM^" << p << "=\n" << mat.power(p).to_string() << '\n';

		} else if (choice == 'R') {
			cout << "Input the number of rows & columns for the matrix e.g. 2 3\n>>> ";
			getline(cin, dim_str_mat_a);
			vector<int> dims = get_dims_from_str(dim_str_mat_a);
			int rows = dims[0], cols = dims[1];

			cout << "Input matrix M (" << (rows * cols) << " entries) e.g. " << get_example(rows * cols) << "\n>>> ";
			getline(cin, mat_a_str);

			Matrix mat(grid_from_line(mat_a_str, rows, cols));

			cout << "\nRREF(M) =\n" << mat.rref().to_string() << '\n';

		} else if (choice == 'I') {
			cout << "Input the size of the square matrix e.g. 2\n>>> ";
			getline(cin, dim_str_mat_a);
			int size = stoi(dim_str_mat_a);

			cout << "Input matrix M (" << (size * size) << " entries) e.g. " << get_example(size * size) << "\n>>> ";
			getline(cin, mat_a_str);

			Matrix mat(grid_from_line(mat_a_str, size, size));

			cout << "\nComatrix =\n" << mat.comatrix().to_string() << '\n';

			long double det = mat.determinant();
			if (det == 0)
				cout << "No inverse (determinant = 0)\n";
			else
				cout << "Inverse:\n" << mat.inverse().to_string() << "\nDeterminant: " << det << "\n\n";

		} else if (choice == 'G') {
			cout << "Enter 1 for translation, 2 for enlargement, 3 for reflection, or 4 for rotation\n>>> ";
			string geom_choice;
			getline(cin, geom_choice);

			cout << "How many vertices?\n>>> ";
			string num_verts_str;
			getline(cin, num_verts_str);
			int num_verts = stoi(num_verts_str);

			vector<vector<long double>> coords(num_verts);
			string str;
			for (int i = 0; i < num_verts; i++) {
				cout << "Input x,y coords " << (i + 1) << '/' << num_verts << " e.g. " << get_example(2) << "\n>>> ";
				getline(cin, str);
				coords[i] = vector_from_line(str);
			}

			Matrix mat(coords);

			if (geom_choice == "1") {
				cout << "Input the change in x and change in y e.g. " << get_example(2) << "\n>>> ";
				getline(cin, str);
				vector<long double> v = vector_from_line(str);
				long double dx = v[0], dy = v[1];
				cout << "\nResultant coords:\n" << mat.translate(dx, dy).to_string() << '\n';
			} else if (geom_choice == "2") {
				cout << "Input the enlargement factor and x, y e.g. 2 3 4\n>>> ";
				getline(cin, str);
				vector<long double> v = vector_from_line(str);
				long double k = v[0], x = v[1], y = v[2];
				cout << "\nResultant coords:\n" << mat.enlarge(k, x, y).to_string() << '\n';
			} else if (geom_choice == "3") {
				cout << "Input m, c for reflection line y = mx + c e.g. 2 3\n>>> ";
				getline(cin, str);
				vector<long double> v = vector_from_line(str);
				long double m = v[0], c = v[1];
				cout << "\nResultant coords:\n" << mat.reflect(m, c).to_string() << '\n';
			} else if (geom_choice == "4") {
				cout << "Input the clockwise rotation angle (deg) and x, y e.g. 90 0 2\n>>> ";
				getline(cin, str);
				vector<long double> v = vector_from_line(str);
				long double a = v[0], x = v[1], y = v[2];
				cout << "\nResultant coords:\n" << mat.rotate(a, x, y).to_string() << '\n';
			}

		} else if (choice == 'X') {
			break;
		}
	}

	return 0;
}
