/*
Noughts and Crosses player using minimax algorithm with alpha-beta pruning

Author: Sam Barba
Created 08/02/2022
*/

#include <SFML/Graphics.hpp>

using std::string;


const string AI = "x";
const string HUMAN = "o";
const string TIE = "t";
const string NONE = " ";
const int BOARD_SIZE = 3;
const int CELL_SIZE = 120;
const int GRID_OFFSET = 80;
const int WINDOW_SIZE = BOARD_SIZE * CELL_SIZE + 2 * GRID_OFFSET;

string board[BOARD_SIZE][BOARD_SIZE];
string statusText;
int squaresLeft;
sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "Noughts and Crosses", sf::Style::Close);
sf::Font font;


void placePlayer(const int y, const int x, const string p) {
	board[y][x] = p;
	squaresLeft--;
}


void removePlayer(const int y, const int x) {
	board[y][x] = NONE;
	squaresLeft++;
}


bool isTie() {
	return squaresLeft == 0;
}


string findWinner() {
	// Check rows and columns
	for (int y = 0; y < 3; y++) {
		if (board[y][0] != NONE && board[y][0] == board[y][1] && board[y][1] == board[y][2]) return board[y][0];
		if (board[0][y] != NONE && board[0][y] == board[1][y] && board[1][y] == board[2][y]) return board[0][y];
	}

	// Check diagonals
	if (board[0][0] != NONE && board[0][0] == board[1][1] && board[1][1] == board[2][2]) return board[1][1];
	if (board[2][0] != NONE && board[2][0] == board[1][1] && board[1][1] == board[0][2]) return board[1][1];

	return isTie() ? TIE : NONE;
}


float minimax(const bool isMaximising, const float depth, float alpha, float beta) {
	if (findWinner() == AI) return 1.f;
	if (findWinner() == HUMAN) return -1.f;
	if (isTie()) return 0.f;

	float score = isMaximising ? -2.f : 2.f;

	for (int y = 0; y < 3; y++) {
		for (int x = 0; x < 3; x++) {
			if (board[y][x] == NONE) {
				if (isMaximising) {
					placePlayer(y, x, AI);
					score = std::max(score, minimax(false, depth + 1, alpha, beta));
					alpha = std::max(alpha, score);
				} else {
					placePlayer(y, x, HUMAN);
					score = std::min(score, minimax(true, depth + 1, alpha, beta));
					beta = std::min(beta, score);
				}
				removePlayer(y, x);
				if (beta <= alpha) return score / depth;
			}
		}
	}

	// Prefer shallower results over deeper results
	return score / depth;
}


void makeBestAIMove() {
	float bestScore = -2.f;
	int bestY, bestX;

	for (int y = 0; y < BOARD_SIZE; y++) {
		for (int x = 0; x < BOARD_SIZE; x++) {
			if (board[y][x] == NONE) {
				placePlayer(y, x, AI);
				float score = minimax(false, 1, -2, 2);
				removePlayer(y, x);
				if (score > bestScore) {
					bestScore = score;
					bestY = y;
					bestX = x;
				}
			}
		}
	}

	placePlayer(bestY, bestX, AI);
	string result = findWinner();
	if (result == AI) statusText = "AI wins! Click to reset";
	if (result == TIE) statusText = "It's a tie! Click to reset";
	if (result == NONE) statusText = "Your turn (o)";
}


void draw() {
	window.clear(sf::Color(20, 20, 20));

	sf::RectangleShape statusLblArea(sf::Vector2f(WINDOW_SIZE, GRID_OFFSET));
	statusLblArea.setPosition(0, 0);
	statusLblArea.setFillColor(sf::Color(20, 20, 20));
	window.draw(statusLblArea);

	sf::Text text(statusText, font, 16);
	sf::FloatRect textRect = text.getLocalBounds();
	text.setOrigin(int(textRect.left + textRect.width / 2), int(textRect.top + textRect.height / 2));
	text.setPosition(WINDOW_SIZE / 2, GRID_OFFSET / 2);
	text.setFillColor(sf::Color::White);
	window.draw(text);

	for (int y = 0; y < BOARD_SIZE; y++) {
		for (int x = 0; x < BOARD_SIZE; x++) {
			string token = board[y][x];
			if (token == NONE) continue;

			sf::Text cellText(token, font, 140);
			cellText.setPosition(int(float(x + 0.19f) * CELL_SIZE + GRID_OFFSET), int(float(y - 0.37f) * CELL_SIZE + GRID_OFFSET));
			cellText.setFillColor(token == AI ? sf::Color(220, 20, 20) : sf::Color(20, 120, 220));
			window.draw(cellText);
		}
	}

	// Grid lines
	for (int i = 0; i < 4; i++) {
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

	window.display();
}


void handleMouseClick(const int mouseX, const int mouseY) {
	int y = (mouseY - GRID_OFFSET) / CELL_SIZE;
	int x = (mouseX - GRID_OFFSET) / CELL_SIZE;
	if (y < 0 || y >= BOARD_SIZE || x < 0 || x >= BOARD_SIZE || board[y][x] != NONE || findWinner() != NONE)
		return;

	placePlayer(y, x, HUMAN);

	string result = findWinner();
	// No point checking if human wins...
	if (result == TIE) statusText = "It's a tie! Click to reset";
	if (result == NONE) statusText = "AI's turn (x)";

	draw();

	if (result == NONE) {  // AI's turn
		sf::sleep(sf::seconds(1));
		makeBestAIMove();
		draw();
	}
}


int main() {
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");
	fill(&board[0][0], &board[0][0] + sizeof(board) / sizeof(board[0][0]), NONE);
	statusText = "Your turn (or 'A' to make AI go first)";
	squaresLeft = 9;

	draw();
	sf::Event event;

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::MouseButtonPressed:
					if (event.mouseButton.button == sf::Mouse::Left) {
						if (findWinner() != NONE) {  // Click to reset if game over
							fill(&board[0][0], &board[0][0] + sizeof(board) / sizeof(board[0][0]), NONE);
							statusText = "Your turn (or 'A' to make AI go first)";
							squaresLeft = 9;
							draw();
						} else {
							sf::Vector2i mousePos = sf::Mouse::getPosition(window);
							handleMouseClick(mousePos.x, mousePos.y);
						}
					}
					break;
				case sf::Event::KeyPressed:
					if (event.key.code == sf::Keyboard::A) {  // Make AI play first
						if (squaresLeft == 9) {  // If no moves have been played yet
							makeBestAIMove();
							draw();
						}
					}
					break;
			}
		}
	}

	return 0;
}
