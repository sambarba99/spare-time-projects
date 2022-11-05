/*
Tic Tac Toe player using minimax algorithm with alpha-beta pruning

Author: Sam Barba
Created 08/02/2022
*/

#include "game.h"

int main() {
	char choice;
	cout << "First move for AI? (Y/N)\n>>> ";
	cin >> choice;

	Game tictactoe;
	tictactoe.play(toupper(choice) == 'Y');
}
