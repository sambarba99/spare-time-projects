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
std::vector<pair<int, int>> yx_given;  // Store coords of numbers that are already given
int num_backtracks;
sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "Sudoku Solver", sf::Style::Close);
sf::Font font;


void draw_grid(const string& status) {
	window.clear(sf::Color(20, 20, 20));

	sf::RectangleShape status_lbl_area(sf::Vector2f(WINDOW_SIZE, GRID_OFFSET));
	status_lbl_area.setPosition(0, 0);
	status_lbl_area.setFillColor(sf::Color(20, 20, 20));
	window.draw(status_lbl_area);

	sf::Text text(status, font, 16);
	sf::FloatRect text_rect = text.getLocalBounds();
	text.setOrigin(int(text_rect.left + text_rect.width / 2), int(text_rect.top + text_rect.height / 2));
	text.setPosition(WINDOW_SIZE / 2, GRID_OFFSET / 2);
	text.setFillColor(sf::Color::White);
	window.draw(text);

	for (int y = 0; y < BOARD_SIZE; y++) {
		for (int x = 0; x < BOARD_SIZE; x++) {
			string str_val = board[y][x] ? to_string(board[y][x]) : "";
			sf::Text cell_text(str_val, font, 22);
			// Draw already given numbers as green
			pair<int, int> temp = {y, x};
			bool is_given = find(yx_given.begin(), yx_given.end(), temp) != yx_given.end();
			cell_text.setPosition(int(float(x + 0.37f) * CELL_SIZE + GRID_OFFSET), int(float(y + 0.22f) * CELL_SIZE + GRID_OFFSET));
			cell_text.setFillColor(is_given ? sf::Color(0, 140, 0) : sf::Color(220, 220, 220));
			window.draw(cell_text);
		}
	}

	// Thin grid lines
	for (int i = 0; i < 10; i++) {
		sf::Vertex line_horiz[] = {
			sf::Vertex(sf::Vector2f(GRID_OFFSET, GRID_OFFSET + i * CELL_SIZE), sf::Color(220, 220, 220)),
			sf::Vertex(sf::Vector2f(GRID_OFFSET + BOARD_SIZE * CELL_SIZE, GRID_OFFSET + i * CELL_SIZE), sf::Color(220, 220, 220))
		};
		sf::Vertex line_vert[] = {
			sf::Vertex(sf::Vector2f(GRID_OFFSET + i * CELL_SIZE, GRID_OFFSET), sf::Color(220, 220, 220)),
			sf::Vertex(sf::Vector2f(GRID_OFFSET + i * CELL_SIZE, GRID_OFFSET + BOARD_SIZE * CELL_SIZE), sf::Color(220, 220, 220))
		};
		window.draw(line_horiz, 2, sf::Lines);
		window.draw(line_vert, 2, sf::Lines);
	}

	// Thick grid lines
	for (int i = 3; i < BOARD_SIZE; i += 3) {
		sf::RectangleShape line_horiz(sf::Vector2f(BOARD_SIZE * CELL_SIZE, 5));
		sf::RectangleShape line_vert(sf::Vector2f(5, BOARD_SIZE * CELL_SIZE));
		line_horiz.setPosition(GRID_OFFSET, GRID_OFFSET + CELL_SIZE * i);
		line_vert.setPosition(GRID_OFFSET + CELL_SIZE * i, GRID_OFFSET);
		line_horiz.setFillColor(sf::Color(220, 220, 220));
		line_vert.setFillColor(sf::Color(220, 220, 220));
		window.draw(line_horiz);
		window.draw(line_vert);
	}

	window.display();
}


bool is_full() {
	for (int y = 0; y < BOARD_SIZE; y++)
		for (int x = 0; x < BOARD_SIZE; x++)
			if (!board[y][x])
				return false;

	return true;
}


pair<int, int> find_free_square() {
	for (int y = 0; y < BOARD_SIZE; y++)
		for (int x = 0; x < BOARD_SIZE; x++)
			if (!board[y][x])
				return {y, x};

	throw std::exception();  // Shouldn't ever get here (if there are no free squares, the sudoku is complete)
}


bool is_legal(const int n, const int y, const int x) {
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
			if (board[i][j] == n)
				return false;

	return true;
}


void solve() {
	if (is_full()) return;

	pair<int, int> free_square = find_free_square();
	int y = free_square.first, x = free_square.second;
	for (int n = 1; n <= 9; n++)
		if (is_legal(n, y, x)) {
			board[y][x] = n;
			draw_grid("Solving (" + to_string(num_backtracks) + " backtracks)");
			solve();
		}

	if (is_full()) return;

	// If we're here, no numbers were legal
	// So the previous attempt in the loop must be invalid
	// So we reset the square in order to backtrack, so next number is tried
	board[y][x] = 0;
	num_backtracks++;
	draw_grid("Solving (" + to_string(num_backtracks) + " backtracks)");
}


void await_click() {
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
		for (const auto& [level, flat_board] : PUZZLES) {
			yx_given.clear();
			num_backtracks = 0;

			for (int i = 0; i < 81; i++) {
				int row = i / BOARD_SIZE, col = i % BOARD_SIZE;
				board[row][col] = flat_board[i] - '0';
				if (board[row][col])
					yx_given.emplace_back(row, col);
			}

			draw_grid("Level: " + level + " (click to solve)");
			await_click();
			high_resolution_clock::time_point start = high_resolution_clock::now();
			solve();
			high_resolution_clock::time_point finish = high_resolution_clock::now();
			auto millis = duration_cast<milliseconds>(finish - start);
			draw_grid("Solved (" + to_string(num_backtracks) + " backtracks, " + to_string(millis.count()) + "ms) - click for next puzzle");
			await_click();
		}
	}

	return 0;
}
