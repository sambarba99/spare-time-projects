/*
Random walk generator

Author: Sam Barba
Created 18/10/2024
*/

#include <iostream>
#include <random>
#include <SFML/Graphics.hpp>


const int NUM_STEPS = 1e7;
const int BORDER = 10;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dist(0, 3);


int main() {
	// 1. Generate a walk

	std::cout << "Walking...\n";

	std::vector<std::pair<int, int>> walkCoords;
	int y = 0, x = 0;
	int dir;
	int minY = NUM_STEPS, minX = NUM_STEPS, maxY = 0, maxX = 0;

	for (int i = 0; i <= NUM_STEPS; i++) {
		walkCoords.push_back({y, x});

		dir = dist(gen);

		if (dir == 0) y--;  // North
		else if (dir == 1) x++;  // East
		else if (dir == 2) y++;  // South
		else x--;  // West

		// Keep track of min/max x/y for normalisation later
		if (y < minY) minY = y;
		if (y > maxY) maxY = y;
		if (x < minX) minX = x;
		if (x > maxX) maxX = x;
	}

	// 2. Normalise the walk coords

	for (auto& coord : walkCoords) {
		coord.first -= minY;
		coord.second -= minX;
	}

	std::cout << "Generating image...\n";

	// 3. Count coord visit frequencies

	std::map<std::pair<int, int>, int> visitCount;
	for (const auto& coord : walkCoords)
		visitCount[coord]++;

	int maxNumVisits = 0;
	for (const auto& entry : visitCount)
		if (entry.second > maxNumVisits)
			maxNumVisits = entry.second;

	// 4. Create image based on normalised coords (making frequently visited coords brighter)

	int pathWidth = maxX - minX + 1;
	int pathHeight = maxY - minY + 1;

	sf::Image image;
	image.create(pathWidth + 2 * BORDER, pathHeight + 2 * BORDER);

	for (const auto& coord : walkCoords) {
		int b = float(visitCount[coord]) / maxNumVisits * 235 + 20;  // Brightness (20 - 255)
		image.setPixel(coord.second + BORDER, coord.first + BORDER, sf::Color(b, b, b));
	}
	image.setPixel(walkCoords[0].second + BORDER, walkCoords[1].first + BORDER, sf::Color::Red);

	image.saveToFile("./rand_walk.png");

	std::cout << "Image saved at ./rand_walk.png";

	return 0;
}
