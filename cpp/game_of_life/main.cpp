/*
Conway's Game of Life

Controls:
	1: preset pattern 1 (glider gun)
	2: preset pattern 2 (R pentomino)
	D: toggle delay after each step (100ms)
	R: randomise cells
	Space: play/pause

Author: Sam Barba
Created 14/11/2022
*/

#include <random>
#include <SFML/Graphics.hpp>

using std::pair;
using std::vector;


const int ROWS = 170;
const int COLS = 300;
const int CELL_SIZE = 5;
const int LABEL_HEIGHT = 30;

const vector<pair<int, int>> GLIDER_GUN = {{85, 132}, {85, 133}, {86, 132}, {86, 133}, {83, 144}, {83, 145}, {84, 143},
	{84, 147}, {85, 142}, {85, 148}, {86, 142}, {86, 146}, {86, 148}, {86, 149}, {87, 142}, {87, 148}, {88, 143},
	{88, 147}, {89, 144}, {89, 145}, {83, 152}, {83, 153}, {84, 152}, {84, 153}, {85, 152}, {85, 153}, {82, 154},
	{86, 154}, {81, 156}, {82, 156}, {86, 156}, {87, 156}, {83, 166}, {83, 167}, {84, 166}, {84, 167}};

const vector<pair<int, int>> R_PENTOMINO = {{83, 149}, {83, 150}, {84, 148}, {84, 149}, {85, 149}};

bool grid[ROWS][COLS];  // True = alive
bool running = true;
bool delay = true;
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(0, 1);
sf::RenderWindow window(sf::VideoMode(COLS * CELL_SIZE, ROWS * CELL_SIZE + LABEL_HEIGHT), "Game of Life", sf::Style::Close);
sf::Font font;


void randomiseLiveCells() {
	for (int y = 0; y < ROWS; y++)
		for (int x = 0; x < COLS; x++)
			grid[y][x] = dist(gen) < 0.125;  // 1 in 8 chance of being alive
}


void setPattern(const vector<pair<int, int>>& pattern) {
	std::fill(&grid[0][0], &grid[0][0] + sizeof(grid) / sizeof(grid[0][0]), false);
	for (const pair<int, int>& coords : pattern)
		grid[coords.first][coords.second] = true;
}


int countLiveNeighbours(const int y, const int x) {
	int n = 0;
	for (int check_y = y - 1; check_y <= y + 1; check_y++) {
		for (int check_x = x - 1; check_x <= x + 1; check_x++) {
			if (0 <= check_y && check_y < ROWS && 0 <= check_x && check_x < COLS)
				if (grid[check_y][check_x])
					n++;
		}
	}

	return grid[y][x] ? n - 1 : n;
}


void updateGrid() {
	bool nextGenGrid[ROWS][COLS] = {{false}};
	int n;

	for (int y = 0; y < ROWS; y++) {
		for (int x = 0; x < COLS; x++) {
			n = countLiveNeighbours(y, x);

			if (n < 2 || n > 3) nextGenGrid[y][x] = false;
			else if (n == 3) nextGenGrid[y][x] = true;
			else nextGenGrid[y][x] = grid[y][x];
		}
	}

	std::copy(&nextGenGrid[0][0], &nextGenGrid[0][0] + ROWS * COLS , &grid[0][0]);
}


void draw() {
	window.clear(sf::Color(40, 40, 40));

	sf::RectangleShape lblArea(sf::Vector2f(COLS * CELL_SIZE, LABEL_HEIGHT));
	lblArea.setPosition(0, 0);
	lblArea.setFillColor(sf::Color::Black);
	window.draw(lblArea);

	sf::Text text("1: preset pattern 1 (glider gun)  |  2: preset pattern 2 (R pentomino)  |  D: toggle 100ms delay  |  R: randomise cells  |  Space: play/pause", font, 16);
	sf::FloatRect textRect = text.getLocalBounds();
	text.setOrigin(int(textRect.left + textRect.width / 2), int(textRect.top + textRect.height / 2));
	text.setPosition(CELL_SIZE * COLS / 2, LABEL_HEIGHT / 2);
	text.setFillColor(sf::Color::White);
	window.draw(text);

	for (int y = 0; y < ROWS; y++) {
		for (int x = 0; x < COLS; x++) {
			if (grid[y][x]) {
				sf::RectangleShape rect(sf::Vector2f(CELL_SIZE, CELL_SIZE));
				rect.setPosition(x * CELL_SIZE, y * CELL_SIZE + LABEL_HEIGHT);
				rect.setFillColor(sf::Color(220, 140, 0));
				window.draw(rect);
			}
		}
	}

	window.display();
}


int main() {
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");
	randomiseLiveCells();
	sf::Event event;

	while (window.isOpen()) {
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
			if (delay) sf::sleep(sf::milliseconds(100));
		}
	}

	return 0;
}
