/*
Sudoku solver using Depth-First Search and backtracking

Author: Sam Barba
Created 07/09/2021
*/

#include <chrono>
#include <SFML/Graphics.hpp>

using namespace std::chrono;
using std::pair;
using std::string;
using std::to_string;


const int BOARD_SIZE = 9;
const int CELL_SIZE = 50;
const int GRID_OFFSET = 75;
const int WINDOW_SIZE = BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET;

// Puzzles in ascending order of difficulty
const std::vector<pair<string, string>> PUZZLES = {
	{"blank", "000000000000000000000000000000000000000000000000000000000000000000000000000000000"},
	{"easy", "000000000010020030000000000000000000040050060000000000000000000070080090000000000"},
	{"medium", "100000000020000000003000000000400000000050000000006000000000700000000080000000009"},
	{"hard", "120000034500000006000000000000070000000891000000020000000000000300000005670000089"},
	{"insane", "800000000003600000070090200050007000000045700000100030001000068008500010090000400"}
};

int board[BOARD_SIZE][BOARD_SIZE];
std::vector<pair<int, int>> yxGiven;  // Store coords of numbers that are already given
int numBacktracks;
sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "Sudoku Solver", sf::Style::Close);
sf::Font font;


void drawGrid(const string status) {
	window.clear(sf::Color(20, 20, 20));

	sf::RectangleShape statusLblArea(sf::Vector2f(WINDOW_SIZE, GRID_OFFSET));
	statusLblArea.setPosition(0, 0);
	statusLblArea.setFillColor(sf::Color(20, 20, 20));
	window.draw(statusLblArea);

	sf::Text text(status, font, 16);
	sf::FloatRect textRect = text.getLocalBounds();
	text.setOrigin(int(textRect.left + textRect.width / 2), int(textRect.top + textRect.height / 2));
	text.setPosition(WINDOW_SIZE / 2, GRID_OFFSET / 2);
	text.setFillColor(sf::Color::White);
	window.draw(text);

	for (int y = 0; y < BOARD_SIZE; y++) {
		for (int x = 0; x < BOARD_SIZE; x++) {
			string strVal = board[y][x] ? to_string(board[y][x]) : "";
			sf::Text cellText(strVal, font, 22);
			// Draw already given numbers as green
			pair<int, int> temp = {y, x};
			bool isGiven = find(yxGiven.begin(), yxGiven.end(), temp) != yxGiven.end();
			cellText.setPosition(int(float(x + 0.37f) * CELL_SIZE + GRID_OFFSET), int(float(y + 0.22f) * CELL_SIZE + GRID_OFFSET));
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
	for (int y = 0; y < BOARD_SIZE; y++)
		for (int x = 0; x < BOARD_SIZE; x++)
			if (!board[y][x]) return false;

	return true;
}


pair<int, int> findFreeSquare() {
	for (int y = 0; y < BOARD_SIZE; y++)
		for (int x = 0; x < BOARD_SIZE; x++)
			if (!board[y][x]) return {y, x};

	return {-1, -1};
}


bool isLegal(const int n, const int y, const int x) {
	// Top-left coords of big square
	int by = y - (y % 3);
	int bx = x - (x % 3);

	// Check row and column
	for (int i = 0; i < BOARD_SIZE; i++) {
		if (board[y][i] == n) return false;
		if (board[i][x] == n) return false;
	}

	// Check big square
	for (int i = by; i < by + 3; i++)
		for (int j = bx; j < bx + 3; j++)
			if (board[i][j] == n) return false;

	return true;
}


void solve() {
	if (isFull()) return;

	pair<int, int> freeSquare = findFreeSquare();
	int y = freeSquare.first, x = freeSquare.second;
	for (int n = 1; n <= 9; n++) {
		if (isLegal(n, y, x)) {
			board[y][x] = n;
			drawGrid("Solving (" + to_string(numBacktracks) + " backtracks)");
			solve();
		}
	}

	if (isFull()) return;

	// If we're here, no numbers were legal
	// So the previous attempt in the loop must be invalid
	// So we reset the square in order to backtrack, so next number is tried
	board[y][x] = 0;
	numBacktracks++;
	drawGrid("Solving (" + to_string(numBacktracks) + " backtracks)");
}


void waitForClick() {
	sf::Event event;

	while (true) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					throw std::exception();
				case sf::Event::MouseButtonPressed:
					return;
			}
		}
	}
}


int main() {
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");

	while (true) {
		for (const auto& item : PUZZLES) {
			yxGiven.clear();
			numBacktracks = 0;

			for (int y = 0; y < 81; y++) {
				int row = y / BOARD_SIZE, col = y % BOARD_SIZE;
				board[row][col] = item.second[y] - '0';
				if (board[row][col])
					yxGiven.push_back({row, col});
			}

			drawGrid("Level: " + item.first + " (click to solve)");
			waitForClick();
			high_resolution_clock::time_point start = high_resolution_clock::now();
			solve();
			high_resolution_clock::time_point finish = high_resolution_clock::now();
			auto millis = duration_cast<milliseconds>(finish - start);
			drawGrid("Solved (" + to_string(numBacktracks) + " backtracks, " + to_string(millis.count()) + "ms) - click for next puzzle");
			waitForClick();
		}
	}

	return 0;
}
