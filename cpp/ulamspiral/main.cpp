/*
Drawing Ulam's spiral (of primes)

Author: Sam Barba
Created 15/11/2022
*/

#include <cmath>
#include <SFML/Graphics.hpp>
#include <vector>

using std::vector;

const int GRID_SIZE = 999;  // Ensure odd
const int CELL_SIZE = 1;

sf::RenderWindow window(sf::VideoMode(GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE), "Ulam's spiral", sf::Style::Close);

vector<bool> primesLessThan(const int n) {
	// Sieve of Eratosthenes

	vector<bool> isPrime(n, true);
	isPrime[0] = false;
	isPrime[1] = false;
	for (int i = 2; i <= sqrt(n) + 1; i++) {
		if (isPrime[i])
			for (int j = i * i; j < n; j += i)
				isPrime[j] = false;
	}

	return isPrime;
}

void draw() {
	int x = (GRID_SIZE * CELL_SIZE) / 2;
	int y = x;
	int state = 0, nSteps = 1, turnCounter = 1;

	sf::Font font;
	font.loadFromFile("C:\\Windows\\Fonts\\consola.ttf");

	int lim = GRID_SIZE * GRID_SIZE + 1;
	vector<bool> isPrime = primesLessThan(lim);

	for (int n = 1; n < lim; n++) {
		sf::RectangleShape rect(sf::Vector2f(CELL_SIZE, CELL_SIZE));
		rect.setPosition(x - CELL_SIZE / 2.f, y - CELL_SIZE / 2.f);
		rect.setFillColor(isPrime[n] ? sf::Color::Red : sf::Color::Black);
		window.draw(rect);

		switch (state) {
			case 0:
				x += CELL_SIZE;
				break;
			case 1:
				y -= CELL_SIZE;
				break;
			case 2:
				x -= CELL_SIZE;
				break;
			case 3:
				y += CELL_SIZE;
				break;
		}

		if (n % nSteps == 0) {
			state = ++state % 4;
			turnCounter++;
			if (turnCounter % 2 == 0) nSteps++;
		}
	}

	window.display();
}

int main() {
	draw();

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
			}
		}
	}
}
