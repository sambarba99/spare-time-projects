#ifndef GAME
#define GAME

#include <algorithm>
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>

using std::pair;
using std::string;
using std::to_string;
using std::vector;

const sf::Color BACKGROUND = sf::Color(20, 20, 20);
const sf::Color CELL_UNCLICKED = sf::Color(80, 80, 80);
const sf::Color CELL_FLAGGED = sf::Color(255, 160, 0);
const sf::Color MINE_WON = sf::Color(0, 144, 0);
const sf::Color MINE_LOST = sf::Color(255, 20, 20);

struct Cell {
	bool isMine;
	bool isFlagged;
	bool isRevealed;
	sf::Color colour;
	string text;
};

class Game {
	public:
		int rows;
		int cols;
		int nMines;
		int flagsUsedTotal;
		int flagsUsedCorrectly;
		bool gameOver;
		bool doneFirstClick;
		vector<vector<Cell>> grid;
		string status;

		Game(const int rows, const int cols, const int nMines) {
			this->rows = rows;
			this->cols = cols;
			this->nMines = nMines;
			setup();
		}

		void setup() {
			vector<vector<Cell>> tempGrid(rows, vector<Cell>(cols, {false, false, false, CELL_UNCLICKED, ""}));
			grid = tempGrid;
			doneFirstClick = gameOver = false;
			flagsUsedTotal = flagsUsedCorrectly = 0;
			status = "Flags left: " + to_string(nMines);
		}

		void firstClick(const int firstI, const int firstJ) {
			doneFirstClick = true;

			// Get all possible coords and shuffle them
			vector<pair<int, int>> allCoords;
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					if (i == firstI && j == firstJ) continue;  // First clicked cell can't be a mine
					allCoords.push_back({i, j});
				}
			}
			random_shuffle(allCoords.begin(), allCoords.end());

			// Place mines randomly
			for (int i = 0; i < nMines; i++)
				grid[allCoords[i].first][allCoords[i].second].isMine = true;

			// Reveal first clicked cell
			handleMouseClick(firstI, firstJ, true);
		}

		void handleMouseClick(const int i, const int j, const bool isLeftClick) {
			if (gameOver || grid[i][j].isRevealed) return;

			if (isLeftClick && !grid[i][j].isFlagged) {
				if (grid[i][j].isMine) {
					endGame(false);
				} else {
					reveal(i, j, false);
					checkWin();
				}
			} else if (!isLeftClick) {  // Right click (toggle flag)
				if (grid[i][j].isFlagged) {
					flagsUsedTotal--;
					if (grid[i][j].isMine)
						flagsUsedCorrectly--;
					grid[i][j].colour = CELL_UNCLICKED;
					grid[i][j].isFlagged = false;
				} else if (nMines - flagsUsedTotal > 0) {  // If there are flags left to use
					flagsUsedTotal++;
					if (grid[i][j].isMine)
						flagsUsedCorrectly++;
					grid[i][j].colour = CELL_FLAGGED;
					grid[i][j].isFlagged = true;
				}
				checkWin();
			}
		}

		void endGame(const bool won) {
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					reveal(i, j, won);
			gameOver = true;
			status = won ? "YOU WIN! Click to reset." : "GAME OVER. Click to reset.";
		}

		int countNeighbourMines(const int i, const int j) {
			int n = 0;
			for (int checki = i - 1; checki <= i + 1; checki++) {
				for (int checkj = j - 1; checkj <= j + 1; checkj++) {
					if (checki < 0 || checki >= rows || checkj < 0 || checkj >= cols) continue;
					if (grid[checki][checkj].isMine)
						n++;
				}
			}
			if (grid[i][j].isMine) n--;  // Don't include self in neighbour count
			return n;
		}

		void reveal(const int i, const int j, const bool won) {
			if (grid[i][j].isRevealed) return;

			grid[i][j].isRevealed = true;

			if (grid[i][j].isMine) {
				grid[i][j].colour = won ? MINE_WON : MINE_LOST;
			} else {
				grid[i][j].colour = BACKGROUND;
				int n = countNeighbourMines(i, j);
				if (n) {
					grid[i][j].text = to_string(n);
				} else {
					// Recursively reveal cells with no neighbouring mines
					for (int checki = i - 1; checki <= i + 1; checki++) {
						for (int checkj = j - 1; checkj <= j + 1; checkj++) {
							if (checki < 0 || checki >= rows || checkj < 0 || checkj >= cols) continue;
							if (grid[checki][checkj].isFlagged) continue;
							reveal(checki, checkj, won);
						}
					}
				}
			}
		}

		void checkWin() {
			// Win if: all mines are correctly flagged; and all non-mine cells are revealed

			bool allNonMinesRevealed = true;
			for (int i = 0; i < rows; i++) {
				if (!allNonMinesRevealed) break;
				for (int j = 0; j < cols; j++) {
					if (!grid[i][j].isMine && !grid[i][j].isRevealed) {
						allNonMinesRevealed = false;
						break;
					}
				}
			}

			if (flagsUsedCorrectly == nMines && allNonMinesRevealed)
				endGame(true);
			else
				status = "Flags left: " + to_string(nMines - flagsUsedTotal);
		}
};

#endif
