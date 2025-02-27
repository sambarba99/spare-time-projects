/*
Ulam prime spiral generator

Author: Sam Barba
Created 15/11/2022
*/

#include <cmath>
#include <SFML/Graphics.hpp>

using std::vector;


const int GRID_SIZE = 899;  // Ensure odd

sf::RenderWindow window(sf::VideoMode(GRID_SIZE, GRID_SIZE), "Ulam prime spiral", sf::Style::Close);


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
	int x = GRID_SIZE / 2;
	int y = x;
	int state = 0, numSteps = 1, turnCounter = 1;

	int lim = GRID_SIZE * GRID_SIZE + 1;
	vector<bool> isPrime = primesLessThan(lim);
	sf::VertexArray pixels(sf::Points, GRID_SIZE * GRID_SIZE);

	for (int n = 1; n < lim; n++) {
		pixels[n - 1] = sf::Vertex(sf::Vector2f(x, y), isPrime[n] ? sf::Color::Red : sf::Color::Black);

		if (state == 0) x++;
		else if (state == 1) y--;
		else if (state == 2) x--;
		else y++;

		if (n % numSteps == 0) {
			state = ++state % 4;
			turnCounter++;
			if (turnCounter % 2 == 0) numSteps++;
		}
	}

	window.draw(pixels);
	window.display();
}


int main() {
	draw();
	sf::Event event;

	while (window.isOpen())
		while (window.pollEvent(event))
			if (event.type == sf::Event::Closed)
				window.close();

	return 0;
}
