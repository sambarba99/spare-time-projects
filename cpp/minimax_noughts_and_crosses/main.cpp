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
string status_text;
int squares_left;
sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "Noughts and Crosses", sf::Style::Close);
sf::Font font;


void place_player(const int y, const int x, const string& p) {
	board[y][x] = p;
	squares_left--;
}


void remove_player(const int y, const int x) {
	board[y][x] = NONE;
	squares_left++;
}


bool is_tie() {
	return squares_left == 0;
}


string find_winner() {
	// Check rows and columns
	for (int y = 0; y < 3; y++) {
		if (board[y][0] != NONE && board[y][0] == board[y][1] && board[y][1] == board[y][2])
			return board[y][0];
		if (board[0][y] != NONE && board[0][y] == board[1][y] && board[1][y] == board[2][y])
			return board[0][y];
	}

	// Check diagonals
	if (board[0][0] != NONE && board[0][0] == board[1][1] && board[1][1] == board[2][2])
		return board[1][1];
	if (board[2][0] != NONE && board[2][0] == board[1][1] && board[1][1] == board[0][2])
		return board[1][1];

	return is_tie() ? TIE : NONE;
}


float minimax(const bool is_maximising, const float depth, float alpha, float beta) {
	if (find_winner() == AI) return 1.f;
	if (find_winner() == HUMAN) return -1.f;
	if (is_tie()) return 0.f;

	float score = is_maximising ? -2.f : 2.f;

	for (int y = 0; y < 3; y++) {
		for (int x = 0; x < 3; x++) {
			if (board[y][x] == NONE) {
				if (is_maximising) {
					place_player(y, x, AI);
					score = std::max(score, minimax(false, depth + 1, alpha, beta));
					alpha = std::max(alpha, score);
				} else {
					place_player(y, x, HUMAN);
					score = std::min(score, minimax(true, depth + 1, alpha, beta));
					beta = std::min(beta, score);
				}
				remove_player(y, x);
				if (beta <= alpha)
					return score / depth;
			}
		}
	}

	// Prefer shallower results over deeper results
	return score / depth;
}


void make_best_ai_move() {
	float best_score = -2.f;
	int best_y, best_x;

	for (int y = 0; y < BOARD_SIZE; y++) {
		for (int x = 0; x < BOARD_SIZE; x++) {
			if (board[y][x] == NONE) {
				place_player(y, x, AI);
				float score = minimax(false, 1, -2, 2);
				remove_player(y, x);
				if (score > best_score) {
					best_score = score;
					best_y = y;
					best_x = x;
				}
			}
		}
	}

	place_player(best_y, best_x, AI);
	string result = find_winner();
	if (result == AI) status_text = "AI wins! Click to reset";
	if (result == TIE) status_text = "It's a tie! Click to reset";
	if (result == NONE) status_text = "Your turn (o)";
}


void draw() {
	window.clear(sf::Color(20, 20, 20));

	sf::RectangleShape status_lbl_area(sf::Vector2f(WINDOW_SIZE, GRID_OFFSET));
	status_lbl_area.setPosition(0, 0);
	status_lbl_area.setFillColor(sf::Color(20, 20, 20));
	window.draw(status_lbl_area);

	sf::Text text(status_text, font, 16);
	sf::FloatRect text_rect = text.getLocalBounds();
	text.setOrigin(int(text_rect.left + text_rect.width / 2), int(text_rect.top + text_rect.height / 2));
	text.setPosition(WINDOW_SIZE / 2, GRID_OFFSET / 2);
	text.setFillColor(sf::Color::White);
	window.draw(text);

	for (int y = 0; y < BOARD_SIZE; y++) {
		for (int x = 0; x < BOARD_SIZE; x++) {
			string token = board[y][x];
			if (token == NONE) continue;

			sf::Text cell_text(token, font, 140);
			cell_text.setPosition(int(float(x + 0.19f) * CELL_SIZE + GRID_OFFSET), int(float(y - 0.37f) * CELL_SIZE + GRID_OFFSET));
			cell_text.setFillColor(token == AI ? sf::Color(220, 20, 20) : sf::Color(20, 120, 220));
			window.draw(cell_text);
		}
	}

	// Grid lines
	for (int i = 0; i < 4; i++) {
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

	window.display();
}


void handle_click(const int mouse_x, const int mouse_y) {
	int y = (mouse_y - GRID_OFFSET) / CELL_SIZE;
	int x = (mouse_x - GRID_OFFSET) / CELL_SIZE;
	if (y < 0 || y >= BOARD_SIZE || x < 0 || x >= BOARD_SIZE || board[y][x] != NONE || find_winner() != NONE)
		return;

	place_player(y, x, HUMAN);

	string result = find_winner();
	// No point checking if human wins...
	if (result == TIE) status_text = "It's a tie! Click to reset";
	if (result == NONE) status_text = "AI's turn (x)";

	draw();

	if (result == NONE) {  // AI's turn
		sf::sleep(sf::seconds(1));
		make_best_ai_move();
		draw();
	}
}


int main() {
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");
	fill(&board[0][0], &board[0][0] + sizeof(board) / sizeof(board[0][0]), NONE);
	status_text = "Your turn (or 'A' to make AI go first)";
	squares_left = 9;
	sf::Event event;

	draw();

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::MouseButtonPressed:
					if (event.mouseButton.button == sf::Mouse::Left) {
						if (find_winner() != NONE) {  // Click to reset if game over
							fill(&board[0][0], &board[0][0] + sizeof(board) / sizeof(board[0][0]), NONE);
							status_text = "Your turn (or 'A' to make AI go first)";
							squares_left = 9;
							draw();
						} else {
							sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
							handle_click(mouse_pos.x, mouse_pos.y);
						}
					}
					break;
				case sf::Event::KeyPressed:
					if (event.key.code == sf::Keyboard::A) {  // Make AI play first
						if (squares_left == 9) {  // If no moves have been played yet
							make_best_ai_move();
							draw();
						}
					}
					break;
			}
		}
	}

	return 0;
}
