/*
Sudoku solver using Depth-First Search and backtracking

Author: Sam Barba
Created 07/09/2021
*/

#include <algorithm>
#include <chrono>
#include <exception>
#include <map>
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>

using namespace std::chrono;
using std::exception;
using std::find;
using std::map;
using std::pair;
using std::string;
using std::to_string;
using std::vector;

const int BOARD_SIZE = 9;
const int CELL_SIZE = 50;
const int GRID_OFFSET = 75;
const int WINDOW_SIZE = BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET;

// Puzzles in ascending order of difficulty
const vector<pair<string, string>> PUZZLES = {
	{"blank", "000000000000000000000000000000000000000000000000000000000000000000000000000000000"},
	{"easy", "000000000010020030000000000000000000040050060000000000000000000070080090000000000"},
	{"medium", "100000000020000000003000000000400000000050000000006000000000700000000080000000009"},
	{"hard", "120000034500000006000000000000070000000891000000020000000000000300000005670000089"},
	{"insane", "800000000003600000070090200050007000000045700000100030001000068008500010090000400"}
};

int board[BOARD_SIZE][BOARD_SIZE];
vector<pair<int, int>> ijGiven;  // Store coords of numbers that are already given
int nBacktracks;
sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "Sudoku Solver", sf::Style::Close);

void drawGrid(const string status) {
	window.clear(sf::Color(20, 20, 20));

	sf::RectangleShape statusLblArea(sf::Vector2f(WINDOW_SIZE, GRID_OFFSET));
	statusLblArea.setPosition(0, 0);
	statusLblArea.setFillColor(sf::Color(20, 20, 20));
	window.draw(statusLblArea);

	sf::Font font;
	font.loadFromFile("C:\\Windows\\Fonts\\consola.ttf");
	sf::Text text(status, font, 16);
	sf::FloatRect textRect = text.getLocalBounds();
	text.setOrigin(int(textRect.left + textRect.width / 2), int(textRect.top + textRect.height / 2));
	text.setPosition(WINDOW_SIZE / 2, GRID_OFFSET / 2);
	text.setFillColor(sf::Color::White);
	window.draw(text);

	for (int i = 0; i < BOARD_SIZE; i++) {
		for (int j = 0; j < BOARD_SIZE; j++) {
			string strVal = board[i][j] ? to_string(board[i][j]) : "";
			sf::Text cellText(strVal, font, 22);
			// Draw already given numbers as green
			pair<int, int> temp = {i, j};
			bool isGiven = find(ijGiven.begin(), ijGiven.end(), temp) != ijGiven.end();
			cellText.setPosition(int(float(j + 0.37f) * CELL_SIZE + GRID_OFFSET), int(float(i + 0.22f) * CELL_SIZE + GRID_OFFSET));
			cellText.setFillColor(isGiven ? sf::Color(0, 140, 0) : sf::Color(220, 220, 220));
			window.draw(cellText);
		}
	}

	// Thin grid lines
	for (int i = 0; i < 10; i++) {
		sf::Vertex lineHor[] = {
			sf::Vertex(sf::Vector2f(GRID_OFFSET, GRID_OFFSET + i * CELL_SIZE), sf::Color(220, 220, 220)),
			sf::Vertex(sf::Vector2f(GRID_OFFSET + BOARD_SIZE * CELL_SIZE, GRID_OFFSET + i * CELL_SIZE), sf::Color(220, 220, 220))
		};
		sf::Vertex lineVer[] = {
			sf::Vertex(sf::Vector2f(GRID_OFFSET + i * CELL_SIZE, GRID_OFFSET), sf::Color(220, 220, 220)),
			sf::Vertex(sf::Vector2f(GRID_OFFSET + i * CELL_SIZE, GRID_OFFSET + BOARD_SIZE * CELL_SIZE), sf::Color(220, 220, 220))
		};
		window.draw(lineHor, 2, sf::Lines);
		window.draw(lineVer, 2, sf::Lines);
	}

	// Thick grid lines
	for (int i = 3; i < BOARD_SIZE; i += 3) {
		sf::RectangleShape lineHor(sf::Vector2f(BOARD_SIZE * CELL_SIZE, 5));
		sf::RectangleShape lineVer(sf::Vector2f(5, BOARD_SIZE * CELL_SIZE));
		lineHor.setPosition(GRID_OFFSET, GRID_OFFSET + CELL_SIZE * i);
		lineVer.setPosition(GRID_OFFSET + CELL_SIZE * i, GRID_OFFSET);
		lineHor.setFillColor(sf::Color(220, 220, 220));
		lineVer.setFillColor(sf::Color(220, 220, 220));
		window.draw(lineHor);
		window.draw(lineVer);
	}

	window.display();
}

bool isFull() {
	for (int i = 0; i < BOARD_SIZE; i++)
		for (int j = 0; j < BOARD_SIZE; j++)
			if (!board[i][j]) return false;

	return true;
}

pair<int, int> findFreeSquare() {
	for (int i = 0; i < BOARD_SIZE; i++)
		for (int j = 0; j < BOARD_SIZE; j++)
			if (!board[i][j]) return {i, j};

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

	pair<int, int> freeSquare = findFreeSquare();
	int i = freeSquare.first, j = freeSquare.second;
	for (int n = 1; n <= 9; n++) {
		if (isLegal(n, i, j)) {
			board[i][j] = n;
			drawGrid("Solving (" + to_string(nBacktracks) + " backtracks)");
			solve();
		}
	}

	if (isFull()) return;

	// If we're here, no numbers were legal
	// So the previous attempt in the loop must be invalid
	// So we reset the square in order to backtrack, so next number is tried
	board[i][j] = 0;
	nBacktracks++;
	drawGrid("Solving (" + to_string(nBacktracks) + " backtracks)");
}

void waitForClick() {
	while (true) {
		sf::Event event;
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					throw exception();
				case sf::Event::MouseButtonPressed:
					return;
			}
		}
	}
}

int main() {
	while (true) {
		for (pair<string, string> item : PUZZLES) {
			ijGiven.clear();
			nBacktracks = 0;

			for (int i = 0; i < 81; i++) {
				int row = i / BOARD_SIZE, col = i % BOARD_SIZE;
				board[row][col] = item.second[i] - '0';
				if (board[row][col])
					ijGiven.push_back({row, col});
			}

			drawGrid("Level: " + item.first + " (click to solve)");
			waitForClick();
			high_resolution_clock::time_point start = high_resolution_clock::now();
			solve();
			high_resolution_clock::time_point finish = high_resolution_clock::now();
			auto millis = duration_cast<milliseconds>(finish - start);
			drawGrid("Solved (" + to_string(nBacktracks) + " backtracks, " + to_string(millis.count()) + "ms) - click for next puzzle");
			waitForClick();
		}
	}
}
