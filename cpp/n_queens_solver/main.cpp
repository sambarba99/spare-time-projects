/*
N Queens solver using depth-first search

Author: Sam Barba
Created 08/02/2022
*/

#include <chrono>
#include <SFML/Graphics.hpp>

using namespace std::chrono;
using std::to_string;


const int N = 12;
const int BLANK = 0;
const int QUEEN = 1;
const int CELL_SIZE = 30;
const int GRID_OFFSET = 60;
const int WINDOW_SIZE = N * CELL_SIZE + 2 * GRID_OFFSET;

int board[N][N];
int nBacktracks;
sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "N Queens Solver", sf::Style::Close);
sf::Font font;


void drawGrid(const std::string status) {
	window.clear(sf::Color::Black);

	sf::RectangleShape statusLblArea(sf::Vector2f(WINDOW_SIZE, GRID_OFFSET));
	statusLblArea.setPosition(0, 0);
	statusLblArea.setFillColor(sf::Color::Black);
	window.draw(statusLblArea);

	sf::Text text(status, font, 18);
	sf::FloatRect textRect = text.getLocalBounds();
	text.setOrigin(int(textRect.left + textRect.width / 2), int(textRect.top + textRect.height / 2));
	text.setPosition(WINDOW_SIZE / 2, GRID_OFFSET / 2);
	text.setFillColor(sf::Color::White);
	window.draw(text);

	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			sf::RectangleShape square(sf::Vector2f(CELL_SIZE, CELL_SIZE));
			square.setPosition(x * CELL_SIZE + GRID_OFFSET, y * CELL_SIZE + GRID_OFFSET);
			square.setFillColor((y + x) % 2 ? sf::Color(20, 20, 20) : sf::Color(60, 60, 60));
			window.draw(square);

			if (board[y][x] == QUEEN) {
				sf::Text cellText("Q", font, 20);
				cellText.setPosition(int(float(x + 0.26f) * CELL_SIZE + GRID_OFFSET), int(float(y + 0.03f) * CELL_SIZE + GRID_OFFSET));
				cellText.setFillColor(sf::Color(220, 150, 0));
				window.draw(cellText);
			}
		}
	}

	window.display();
}


bool valid(const int row, const int col) {
	// Check if there is a queen above in this column
	for (int y = 0; y < row; y++)
		if (board[y][col]) return false;
	
	// Check upper diagonal on left side
	for (int y = row, x = col; y >= 0 && x >= 0; y--, x--)
		if (board[y][x]) return false;
	
	// Check upper diagonal on right side
	for (int y = row, x = col; y >= 0 && x < N; y--, x++)
		if (board[y][x]) return false;

	return true;
}


bool solve(int row = 0) {
	if (row == N) return true;  // All queens placed

	for (int col = 0; col < N; col++) {
		if (valid(row, col)) {
			board[row][col] = QUEEN;
			// drawGrid("Solving (" + to_string(nBacktracks) + " backtracks)");
			if (solve(row + 1)) return true;
		}

		// Reset square in order to backtrack
		board[row][col] = BLANK;
		nBacktracks++;
		drawGrid("Solving (" + to_string(nBacktracks) + " backtracks)");
	}

	return false;
}


int main() {
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");
	nBacktracks = 0;
	high_resolution_clock::time_point start = high_resolution_clock::now();
	sf::Event event;

	if (solve()) {
		high_resolution_clock::time_point finish = high_resolution_clock::now();
		auto millis = duration_cast<milliseconds>(finish - start);
		drawGrid("Solved (" + to_string(nBacktracks) + " backtracks, " + to_string(millis.count()) + "ms)");
	} else {
		drawGrid("No solution");
	}

	while (window.isOpen())
		while (window.pollEvent(event))
			if (event.type == sf::Event::Closed)
				window.close();

	return 0;
}
