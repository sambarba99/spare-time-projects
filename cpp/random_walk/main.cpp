/*
Random walk generator

Author: Sam Barba
Created 18/10/2024
*/

#include <iostream>
#include <SFML/Graphics.hpp>

using std::cout;
using std::map;
using std::pair;
using std::vector;


const int NUM_STEPS = 1e7;
const int BORDER = 10;


int main() {
	// 1. Generate a walk

	cout << "Walking...\n";

	vector<pair<int, int>> walkCoords;
	int y = 0, x = 0;
	float r;
	int minY = NUM_STEPS, minX = NUM_STEPS, maxY = 0, maxX = 0;

	for (int i = 0; i <= NUM_STEPS; i++) {
		walkCoords.push_back({y, x});

		r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

		if (r < 0.25) y--;  // North
		else if (r < 0.5) x++;  // East
		else if (r < 0.75) y++;  // South
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

	cout << "Generating image...\n";

	// 3. Count coord visit frequencies

	map<pair<int, int>, int> visitCount;
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
		int b = static_cast<int>((static_cast<float>(visitCount[coord]) / maxNumVisits) * 235) + 20;  // Brightness (20 - 255)
		sf::Color colour(b, b, b);
		image.setPixel(coord.second + BORDER, coord.first + BORDER, colour);
	}
	image.setPixel(walkCoords[0].second + BORDER, walkCoords[1].first + BORDER, sf::Color::Red);

	image.saveToFile("./rand_walk.png");

	cout << "Image saved at ./rand_walk.png";
}
