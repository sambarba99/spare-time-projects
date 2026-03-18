/*
Elementary cellular automata demo (click to cycle through rulesets)

Author: Sam Barba
Created 15/11/2022
*/

#include <SFML/Graphics.hpp>

using std::vector;


const int IMG_SIZE = 899;  // Ensure odd
const int RULES[] = {18, 30, 45, 54, 57, 73, 105, 151, 153, 161};  // Interesting rules

sf::RenderWindow window(sf::VideoMode(IMG_SIZE, IMG_SIZE), "", sf::Style::Close);


vector<int> get_ruleset(int n) {
	vector<int> ruleset(8, 0);
	int i = 7;
	while (n) {
		ruleset[i] = n % 2;
		n /= 2;
		i--;
	}

	return ruleset;
}


void generate_plot(const vector<int>& ruleset) {
	vector<int> gen(IMG_SIZE, 0);
	gen[IMG_SIZE / 2] = 1;  // Turn on centre pixel of first generation
	vector<vector<int>> plot(IMG_SIZE, vector<int>(IMG_SIZE, 0));

	for (int y = 0; y < IMG_SIZE; y++) {
		plot[y] = gen;
		vector<int> nextGen(IMG_SIZE, 0);

		for (int x = 0; x < IMG_SIZE; x++) {
			int left = x == 0 ? 0 : gen[x - 1];
			int centre = gen[x];
			int right = x == IMG_SIZE - 1 ? 0 : gen[x + 1];
			nextGen[x] = ruleset[7 - (4 * left + 2 * centre + right)];
		}

		gen = nextGen;
	}

	window.clear(sf::Color(20, 20, 20));
	sf::VertexArray pixels(sf::Points, IMG_SIZE * IMG_SIZE);

	for (int y = 0; y < IMG_SIZE; y++)
		for (int x = 0; x < IMG_SIZE; x++)
			if (plot[y][x])
				pixels[y * IMG_SIZE + x] = sf::Vertex(sf::Vector2f(x, y), sf::Color::White);

	window.draw(pixels);
	window.display();
}


int main() {
	int i = 0;
	generate_plot(get_ruleset(RULES[i]));
	window.setTitle("Elementary Cellular Automata (rule " + std::to_string(RULES[i]) + ")");
	sf::Event event;

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::MouseButtonPressed:
					i++;
					if (i >= sizeof(RULES) / sizeof(int))
						i = 0;
					window.setTitle("Elementary Cellular Automata (rule " + std::to_string(RULES[i]) + ")");
					generate_plot(get_ruleset(RULES[i]));
					break;
			}
		}
	}

	return 0;
}
