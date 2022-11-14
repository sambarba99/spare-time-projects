/*
Minesweeper

Author: Sam Barba
Created 19/11/2022

Controls:
Left-click: reveal a cell
Right-click: flag a cell
*/

#include "game.h"

const int ROWS = 35;
const int COLS = 60;
const int CELL_SIZE = 22;
const int GRID_OFFSET = 60;
const int WINDOW_WIDTH = COLS * CELL_SIZE + 2 * GRID_OFFSET;
const int WINDOW_HEIGHT = ROWS * CELL_SIZE + 2 * GRID_OFFSET;
const int N_MINES = ROWS * COLS * 0.1;  // 10% mines
const sf::Color LABEL_FOREGROUND = sf::Color(220, 220, 220);

sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Minesweeper", sf::Style::Close);

void drawGrid(const vector<vector<Cell>> grid, const string status) {
	window.clear(BACKGROUND);

	sf::RectangleShape statusLblArea(sf::Vector2f(WINDOW_WIDTH, GRID_OFFSET));
	statusLblArea.setPosition(0, 0);
	statusLblArea.setFillColor(BACKGROUND);
	window.draw(statusLblArea);

	sf::Font font;
	font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf");
	sf::Text text(status, font, 18);
	sf::FloatRect textRect = text.getLocalBounds();
	text.setOrigin(int(textRect.left + textRect.width / 2), int(textRect.top + textRect.height / 2));
	text.setPosition(WINDOW_WIDTH / 2, GRID_OFFSET / 2);
	text.setFillColor(LABEL_FOREGROUND);
	window.draw(text);

	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			sf::RectangleShape cell(sf::Vector2f(CELL_SIZE, CELL_SIZE));
			cell.setPosition(j * CELL_SIZE + GRID_OFFSET, i * CELL_SIZE + GRID_OFFSET);
			cell.setFillColor(grid[i][j].colour);
			sf::Text cellText(grid[i][j].text, font, 15);
			cellText.setPosition(int(float(j + 0.3f) * CELL_SIZE + GRID_OFFSET), int(float(i + 0.1f) * CELL_SIZE + GRID_OFFSET));
			cellText.setFillColor(LABEL_FOREGROUND);
			window.draw(cell);
			window.draw(cellText);
		}
	}

	// Grid lines
	for (int i = GRID_OFFSET; i <= ROWS * CELL_SIZE + GRID_OFFSET; i += CELL_SIZE) {
		sf::Vertex line[] = {
			sf::Vertex(sf::Vector2f(GRID_OFFSET, i), BACKGROUND),
			sf::Vertex(sf::Vector2f(COLS * CELL_SIZE + GRID_OFFSET, i), BACKGROUND)
		};
		window.draw(line, 2, sf::Lines);
	}
	for (int j = GRID_OFFSET; j <= COLS * CELL_SIZE + GRID_OFFSET; j += CELL_SIZE) {
		sf::Vertex line[] = {
			sf::Vertex(sf::Vector2f(j, GRID_OFFSET), BACKGROUND),
			sf::Vertex(sf::Vector2f(j, ROWS * CELL_SIZE + GRID_OFFSET), BACKGROUND)
		};
		window.draw(line, 2, sf::Lines);
	}

	window.display();
}

int main() {
	Game game(ROWS, COLS, N_MINES);
	drawGrid(game.grid, game.status);

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::MouseButtonPressed:
					sf::Vector2i mousePos = sf::Mouse::getPosition(window);
					int mouseX = mousePos.x;
					int mouseY = mousePos.y;
					int i = (mouseY - GRID_OFFSET) / CELL_SIZE;
					int j = (mouseX - GRID_OFFSET) / CELL_SIZE;
					if (i < 0 || i >= ROWS || j < 0 || j >= COLS) continue;

					if (event.mouseButton.button == sf::Mouse::Left) {
						if (!game.doneFirstClick)
							game.firstClick(i, j);
						else if (game.gameOver)
							game.setup();
						else
							game.handleMouseClick(i, j, true);
					} else if (event.mouseButton.button == sf::Mouse::Right) {
						if (game.doneFirstClick)
							game.handleMouseClick(i, j, false);
					}

					drawGrid(game.grid, game.status);
					break;
			}
		}
	}
}
