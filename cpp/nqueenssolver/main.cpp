/*
N Queens solver using depth-first search

Author: Sam Barba
Created 08/02/2022
*/

#include <chrono>
#include <iostream>
#include <vector>

using namespace std::chrono;
using std::cout;
using std::vector;

const int N = 24;  // N > 3
const int BLANK = 0;
const int QUEEN = 1;

vector<vector<int>> board(N, vector<int>(N, BLANK));

bool valid(const int row, const int col) {
	// Check if there is a queen above in this column
	for (int i = 0; i < row; i++)
		if (board[i][col]) return false;
	
	// Check upper diagonal on left side
	for (int i = row, j = col; i >= 0 && j >= 0; i--, j--)
		if (board[i][j]) return false;
	
	// Check upper diagonal on right side
	for (int i = row, j = col; i >= 0 && j < N; i--, j++)
		if (board[i][j]) return false;

	return true;
}

bool solve(int row = 0) {
	if (row == N) return true;  // All queens placed

	for (int col = 0; col < N; col++) {
		if (valid(row, col)) {
			board[row][col] = QUEEN;
			if (solve(row + 1)) return true;
		}

		// Reset square in order to backtrack
		board[row][col] = BLANK;
	}

	return false;
}

void printBoard(const int timeTaken) {
	cout << "Solved for N = " << N << " in " << timeTaken << "ms:\n\n";
	for (vector<int> row : board) {
		for (int i : row)
			cout << (i == QUEEN ? " Q" : " -");
		cout << '\n';
	}
}

int main() {
	high_resolution_clock::time_point start = high_resolution_clock::now();
	if (solve()) {
		high_resolution_clock::time_point finish = high_resolution_clock::now();
		auto millis = duration_cast<milliseconds>(finish - start);
		printBoard(millis.count());
	} else {
		cout << "Cannot solve for N = " << N;
	}
}
