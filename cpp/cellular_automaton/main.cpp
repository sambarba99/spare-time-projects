/*
Elementary cellular automaton demo (click to cycle through rulesets)

Author: Sam Barba
Created 15/11/2022
*/

#include <SFML/Graphics.hpp>
#include <vector>

using std::to_string;
using std::vector;

const int IMG_SIZE = 999;  // Ensure odd
const int CELL_SIZE = 1;
const int RULES[] = {18, 30, 45, 54, 57, 73, 105, 151, 153, 161};  // Interesting rules

sf::RenderWindow window(sf::VideoMode(IMG_SIZE * CELL_SIZE, IMG_SIZE * CELL_SIZE), "", sf::Style::Close);

vector<int> getRuleset(int n) {
	vector<int> ruleset(8, 0);
	int i = 7;
	while (n) {
		ruleset[i] = n % 2;
		n /= 2;
		i--;
	}

	return ruleset;
}

void generatePlot(const vector<int> ruleset) {
	vector<int> gen(IMG_SIZE, 0);
	gen[IMG_SIZE / 2] = 1;  // Turn on centre pixel of first generation
	vector<vector<int>> plot(IMG_SIZE, vector<int>(IMG_SIZE, 0));

	for (int i = 0; i < IMG_SIZE; i++) {
		plot[i] = gen;
		vector<int> nextGen(IMG_SIZE, 0);

		for (int j = 0; j < IMG_SIZE; j++) {
			int left = j == 0 ? 0 : gen[j - 1];
			int centre = gen[j];
			int right = j == IMG_SIZE - 1 ? 0 : gen[j + 1];
			nextGen[j] = ruleset[7 - (4 * left + 2 * centre + right)];
		}

		gen = nextGen;
	}

	window.clear(sf::Color(20, 20, 20));

	for (int i = 0; i < IMG_SIZE; i++) {
		for (int j = 0; j < IMG_SIZE; j++) {
			if (!plot[i][j]) continue;

			sf::RectangleShape rect(sf::Vector2f(CELL_SIZE, CELL_SIZE));
			rect.setPosition(j * CELL_SIZE, i * CELL_SIZE);
			rect.setFillColor(sf::Color::White);
			window.draw(rect);
		}
	}

	window.display();
}

int main() {
	int i = 0;
	generatePlot(getRuleset(RULES[i]));
	window.setTitle("Elementary Cellular Automaton (rule " + to_string(RULES[i]) + ")");

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::MouseButtonPressed:
					i = ++i % (sizeof(RULES) / sizeof(int));
					window.setTitle("Elementary Cellular Automaton (rule " + to_string(RULES[i]) + ")");
					generatePlot(getRuleset(RULES[i]));
					break;
			}
		}
	}
}
