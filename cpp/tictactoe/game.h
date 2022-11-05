#ifndef GAME
#define GAME

#include <algorithm>
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::max;
using std::min;
using std::string;

class Game {
	enum class Player : char {
		none = ' ',
		ai = 'X',
		human = 'O'
	};

	struct Move {
		int i = 0;
		int j = 0;
	};

	public:
		Game() {
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
					board[i][j] = Player::none;
		}

		void placePlayer(const int i, const int j, const Player p) {
			board[i][j] = p;
			squaresLeft--;
		}

		void removePlayer(const int i, const int j) {
			board[i][j] = Player::none;
			squaresLeft++;
		}

		void printBoard() {
			cout << '\n';
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					cout << ' ' << static_cast<char>(board[i][j]);
					if (i < 2 && j == 2) cout << "\n---+---+---";
					cout << (j < 2 ? " |" : "\n");
				}
			}
		}

		bool checkWin(Player player) {
			// Check rows and columns
			for (int i = 0; i < 3; i++) {
				if (board[i][0] == player && board[i][1] == player && board[i][2] == player) return true;
				if (board[0][i] == player && board[1][i] == player && board[2][i] == player) return true;
			}

			// Check diagonals
			if (board[0][0] == player && board[1][1] == player && board[2][2] == player) return true;
			if (board[0][2] == player && board[1][1] == player && board[2][0] == player) return true;

			return false;
		}

		bool isTie() {
			return squaresLeft == 0;
		}

		float minimax(const bool isMaximising, const float depth, float alpha, float beta) {
			if (checkWin(Player::ai)) return 1.0;
			if (checkWin(Player::human)) return -1.0;
			if (isTie()) return 0.0;

			float score = isMaximising ? -2.0 : 2.0;

			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					if (board[i][j] == Player::none) {
						if (isMaximising) {
							placePlayer(i, j, Player::ai);
							score = max(score, minimax(false, depth + 1, alpha, beta));
							alpha = max(alpha, score);
						} else {
							placePlayer(i, j, Player::human);
							score = min(score, minimax(true, depth + 1, alpha, beta));
							beta = min(beta, score);
						}
						removePlayer(i, j);
						if (beta <= alpha) return score / depth;
					}
				}
			}

			// Prefer shallower results over deeper results
			return score / depth;
		}

		Move getBestAIMove() {
			float bestScore = -2.0;
			Move move;

			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					if (board[i][j] == Player::none) {
						placePlayer(i, j, Player::ai);
						float score = minimax(false, 1, -2, 2);
						removePlayer(i, j);
						if (score > bestScore) {
							bestScore = score;
							move.i = i;
							move.j = j;
						}
					}
				}
			}

			return move;
		}

		Move getHumanMove() {
			string moveStr;
			Move move;

			while (true) {
				cout << "\nInput move\n>>> ";
				cin >> moveStr;
				move.i = moveStr[0] - '0';
				move.j = moveStr[2] - '0';

				if (move.i < 0 || move.i > 2 || move.j < 0 || move.j > 2) cout << "Coords must be 0-2 and format row,col";
				else if (board[move.i][move.j] != Player::none) cout << "Square already occupied!";
				else return move;
			}

			return move;
		}

		void play(const bool aiFirst) {
			bool humanTurn = !aiFirst, gameOver = false;

			cout << "\n----- Input your move (O) as (row,col) e.g. 0,1 -----\n";

			while (!gameOver) {
				printBoard();

				if (humanTurn) {
					Move humanMove = getHumanMove();
					placePlayer(humanMove.i, humanMove.j, Player::human);

					// No point checking if human wins...
					// if (checkWin(Player::human)) {
						// cout << "\nYou win!\n";
						// gameOver = true;
					// }
				} else {
					Move aiMove = getBestAIMove();
					placePlayer(aiMove.i, aiMove.j, Player::ai);
					cout << "\nAI (X) move: " << aiMove.i << ',' << aiMove.j << '\n';

					if (checkWin(Player::ai)) {
						cout << "\nAI wins!\n";
						gameOver = true;
					}
				}

				if (!gameOver && isTie()) {
					cout << "\nIt's a tie!\n";
					break;
				}

				humanTurn = !humanTurn;
			}

			printBoard();
		}
	
	private:
		int squaresLeft = 9;
		Player board[3][3];
};

#endif
