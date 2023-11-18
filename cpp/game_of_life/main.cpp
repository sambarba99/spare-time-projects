/*
Conway's Game of Life

Author: Sam Barba
Created 14/11/2022

Controls:
1: preset pattern 1 (glider gun)
2: preset pattern 2 (R pentomino)
D: toggle delay after each step (50ms)
R: reset and randomise
Space: play/pause
*/

#include <algorithm>
#include <SFML/Graphics.hpp>
#include <vector>

using std::copy;
using std::fill;
using std::pair;
using std::vector;

const int ROWS = 300;
const int COLS = 600;
const int CELL_SIZE = 3;
const int LABEL_HEIGHT = 40;

const vector<pair<int, int>> GLIDER_GUN = {{135, 275}, {135, 276}, {136, 275}, {136, 276}, {133, 287},
	{133, 288}, {134, 286}, {134, 290}, {135, 285}, {135, 291}, {136, 285}, {136, 289}, {136, 291},
	{136, 292}, {137, 285}, {137, 291}, {138, 286}, {138, 290}, {139, 287}, {139, 288}, {133, 295},
	{133, 296}, {134, 295}, {134, 296}, {135, 295}, {135, 296}, {132, 297}, {136, 297}, {131, 299},
	{132, 299}, {136, 299}, {137, 299}, {133, 309}, {133, 310}, {134, 309}, {134, 310}};

const vector<pair<int, int>> R_PENTOMINO = {{138, 299}, {138, 300}, {139, 298}, {139, 299}, {140, 299}};

bool grid[ROWS][COLS];  // True = alive
bool running = true;
bool delay = true;
sf::RenderWindow window(sf::VideoMode(COLS * CELL_SIZE, ROWS * CELL_SIZE + LABEL_HEIGHT), "Game of Life", sf::Style::Close);

void randomiseLiveCells() {
	float r;
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
			grid[i][j] = r < 0.2;
		}
	}
}

void setPattern(const vector<pair<int, int>> pattern) {
	fill(&grid[0][0], &grid[0][0] + sizeof(grid) / sizeof(grid[0][0]), false);
	for (pair<int, int> coords : pattern)
		grid[coords.first][coords.second] = true;
}

int countLiveNeighbours(const int i, const int j) {
	int n = 0;
	for (int checki = i - 1; checki <= i + 1; checki++) {
		for (int checkj = j - 1; checkj <= j + 1; checkj++) {
			if (0 <= checki && checki < ROWS && 0 <= checkj && checkj < COLS)
				if (grid[checki][checkj])
					n++;
		}
	}

	return grid[i][j] ? n - 1 : n;
}

void updateGrid() {
	bool nextGenGrid[ROWS][COLS] = {{false}};
	int n;

	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			n = countLiveNeighbours(i, j);

			if (n < 2 || n > 3) nextGenGrid[i][j] = false;
			else if (n == 3) nextGenGrid[i][j] = true;
			else nextGenGrid[i][j] = grid[i][j];
		}
	}

	copy(&nextGenGrid[0][0], &nextGenGrid[0][0] + ROWS * COLS , &grid[0][0]);
}

void draw() {
	window.clear(sf::Color(40, 40, 40));

	sf::RectangleShape lblArea(sf::Vector2f(COLS * CELL_SIZE, LABEL_HEIGHT));
	lblArea.setPosition(0, 0);
	lblArea.setFillColor(sf::Color::Black);
	window.draw(lblArea);

	sf::Font font;
	font.loadFromFile("C:\\Windows\\Fonts\\consola.ttf");
	sf::Text text("1: preset pattern 1 (glider gun)  |  2: preset pattern 2 (R pentomino)  |  D: toggle 50ms delay  |  R: reset/randomise  |  Space: play/pause", font, 16);
	sf::FloatRect textRect = text.getLocalBounds();
	text.setOrigin(int(textRect.left + textRect.width / 2), int(textRect.top + textRect.height / 2));
	text.setPosition(CELL_SIZE * COLS / 2, LABEL_HEIGHT / 2);
	text.setFillColor(sf::Color::White);
	window.draw(text);

	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			if (grid[i][j]) {
				sf::RectangleShape rect(sf::Vector2f(CELL_SIZE, CELL_SIZE));
				rect.setPosition(j * CELL_SIZE, i * CELL_SIZE + LABEL_HEIGHT);
				rect.setFillColor(sf::Color(220, 140, 0));
				window.draw(rect);
			}
		}
	}

	window.display();
}

int main() {
	randomiseLiveCells();

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::KeyPressed:
					switch (event.key.code) {
						case sf::Keyboard::Num1:
							setPattern(GLIDER_GUN);
							break;
						case sf::Keyboard::Num2:
							setPattern(R_PENTOMINO);
							break;
						case sf::Keyboard::D:
							delay = !delay;
							break;
						case sf::Keyboard::R:
							randomiseLiveCells();
							delay = running = true;
							break;
						case sf::Keyboard::Space:
							running = !running;
							break;
					}
					break;
			}
		}

		draw();
		if (running) {
			updateGrid();
			if (delay) sf::sleep(sf::milliseconds(50));
		}
	}
}
