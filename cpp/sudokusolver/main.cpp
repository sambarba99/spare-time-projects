/*
Sudoku solver using depth-first search

Author: Sam Barba
Created 07/09/2021
*/

#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std::chrono;
using std::cin;
using std::cout;
using std::map;
using std::string;
using std::vector;

const int BOARD_SIZE = 9;

// Puzzles in ascending order of difficulty
map<string, string> PUZZLES = {
	{"blank", "000000000000000000000000000000000000000000000000000000000000000000000000000000000"},
	{"easy", "000000000010020030000000000000000000040050060000000000000000000070080090000000000"},
	{"medium", "100000000020000000003000000000400000000050000000006000000000700000000080000000009"},
	{"hard", "120000034500000006000000000000070000000891000000020000000000000300000005670000089"},
	{"insane", "800000000003600000070090200050007000000045700000100030001000068008500010090000400"}
};

vector<vector<int>> board(BOARD_SIZE, vector<int>(BOARD_SIZE, 0));

bool isFull() {
	for (vector<int> row : board)
		for (int i : row)
			if (i == 0) return false;
	return true;
}

vector<int> findFreeSquare() {
	for (int i = 0; i < BOARD_SIZE; i++)
		for (int j = 0; j < BOARD_SIZE; j++)
			if (board[i][j] == 0) return {i, j};

	return {-1, -1};
}

bool isLegal(const int n, const int i, const int j) {
	// Top-left coords of big square
	int bi = i - (i % 3);
	int bj = j - (j % 3);

	// Check row and column
	for (int k = 0; k < BOARD_SIZE; k++) {
		if (board[i][k] == n) return false;
		if (board[k][j] == n) return false;
	}

	// Check big square
	for (int k = bi; k < bi + 3; k++)
		for (int l = bj; l < bj + 3; l++)
			if (board[k][l] == n) return false;

	return true;
}

void solve() {
	if (isFull()) return;

	vector<int> freeSquare = findFreeSquare();
	int i = freeSquare[0], j = freeSquare[1];
	for (int n = 1; n <= 9; n++) {
		if (isLegal(n, i, j)) {
			board[i][j] = n;
			solve();
		}
	}

	if (isFull()) return;

	// If we're here, no numbers were legal
	// So the previous attempt in the loop must be invalid
	// So we reset the square in order to backtrack, so next number is tried
	board[i][j] = 0;
}

bool loadBoard(const char choice) {
	string puzzle = "";

	switch (choice) {
		case 'B':
			puzzle = PUZZLES["blank"];
			break;
		case 'E':
			puzzle = PUZZLES["easy"];
			break;
		case 'M':
			puzzle = PUZZLES["medium"];
			break;
		case 'H':
			puzzle = PUZZLES["hard"];
			break;
		case 'I':
			puzzle = PUZZLES["insane"];
			break;
		default:
			return false;
	}

	for (int i = 0; i < 81; i++) {
		int row = i / BOARD_SIZE, col = i % BOARD_SIZE;
		board[row][col] = puzzle[i] - '0';
	}

	return true;
}

void printBoard(const int timeTaken = -1) {
	if (timeTaken == -1) cout << "\nStart:\n\n";
	else cout << "\nSolved in " << timeTaken << "ms:\n\n";

	for (int row = 0; row < BOARD_SIZE; row++) {
		for (int col = 0; col < BOARD_SIZE; col++) {
			int val = board[row][col];
			val == 0 ? cout << "  " : cout << ' ' << val;
			if (col == 2 || col == 5) cout << " |";
		}
		if (row == 2 || row == 5) cout << "\n ------+-------+------";
		cout << '\n';
	}
}

int main() {
	char choice;

	while (true) {
		cout << "Input B/E/M/H/I for blank/easy/medium/hard/insane (or X to exit)\n>>> ";
		cin >> choice;
		choice = toupper(choice);

		if (choice == 'X') break;

		bool success = loadBoard(choice);
		if (success) {
			printBoard();
			high_resolution_clock::time_point start = high_resolution_clock::now();
			solve();
			high_resolution_clock::time_point finish = high_resolution_clock::now();
			auto millis = duration_cast<milliseconds>(finish - start);
			printBoard(millis.count());
			cout << '\n';
		} else {
			cout << "Bad choice\n";
		}
	}
}
