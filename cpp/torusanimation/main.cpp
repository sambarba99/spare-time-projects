/*
Revolving torus animation (press space to toggle animation)

Author: Sam Barba
Created 15/11/2022
*/

#include <algorithm>
#include <cmath>
#include <SFML/Graphics.hpp>
#include <string>

using std::fill;
using std::max;
using std::string;

const int GRID_SIZE = 70;  // Rows, cols
const int CELL_SIZE = 12;

const double CHANGE_THETA = 2 * M_PI / 70.0;
const double CHANGE_PHI = 2 * M_PI / 170.0;

const double R1 = 2.0;  // Radius of torus cross-sectional circle
const double R2 = 5.0;  // Distance from origin to centre of cross-sectional circle

/*
Calculate K1 based on screen size: the maximum x-distance occurs roughly at the edge of the torus,
which is at x = R1 + R2, z = 0. We want that to be displaced 3/8ths of the width of the screen.
*/
const double K2 = 100.0;  // Arbitrary distance from torus to viewer
const double K1 = GRID_SIZE * K2 * 3 / (8 * (R1 + R2));
const string CHARS = ".,-~:;!*=#$@";  // 'Dimmest' to 'brighest' chars

char outputGrid[GRID_SIZE][GRID_SIZE];
sf::RenderWindow window(sf::VideoMode(GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE), "Revolving torus", sf::Style::Close);

void renderTorus() {
	window.clear(sf::Color::Black);

	sf::Font font;
	font.loadFromFile("C:\\Windows\\Fonts\\consola.ttf");

	for (int i = 0; i < GRID_SIZE; i++) {
		for (int j = 0; j < GRID_SIZE; j++) {
			sf::Text text(outputGrid[i][j], font, 14);
			text.setPosition(int(j * CELL_SIZE), int(i * CELL_SIZE));
			text.setFillColor(sf::Color::White);
			window.draw(text);
		}
	}

	window.display();
}

int main() {
	// Revolution amounts about x and z axes (increased after each frame, to revolve the torus)
	double xrev, zrev;
	double sinXrev, cosXrev, sinZrev, cosZrev;
	double sinTheta, cosTheta, sinPhi, cosPhi;
	double zBuffer[GRID_SIZE][GRID_SIZE];
	bool paused = false;

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::KeyPressed:
					if (event.key.code == sf::Keyboard::Space)
						paused = !paused;
					break;
			}
		}

		if (paused) continue;

		sinXrev = sin(xrev);
		cosXrev = cos(xrev);
		sinZrev = sin(zrev);
		cosZrev = cos(zrev);

		fill(&outputGrid[0][0], &outputGrid[0][0] + sizeof(outputGrid) / sizeof(outputGrid[0][0]), ' ');
		fill(&zBuffer[0][0], &zBuffer[0][0] + sizeof(zBuffer) / sizeof(zBuffer[0][0]), 0.0);

		// Theta goes around the cross-sectional circle of the torus
		for (double theta = 0.0; theta < 2 * M_PI; theta += CHANGE_THETA) {
			sinTheta = sin(theta);
			cosTheta = cos(theta);

			// Phi revolves this circle around the y-axis, creating the torus (solid of revolution)
			for (double phi = 0.0; phi < 2 * M_PI; phi += CHANGE_PHI) {
				sinPhi = sin(phi);
				cosPhi = cos(phi);

				// x,y coords before revolution
				double circleX = R2 + R1 * cosTheta;
				double circleY = R1 * sinTheta;

				// 3D coords after revolution
				double x = circleX * (cosZrev * cosPhi + sinXrev * sinZrev * sinPhi) - circleY * cosXrev * sinZrev;
				double y = circleX * (sinZrev * cosPhi - sinXrev * cosZrev * sinPhi) + circleY * cosXrev * cosZrev;
				double z = K2 + cosXrev * circleX * sinPhi + circleY * sinXrev;
				double zr = 1.0 / z;

				// x, y projection (y is negated, as y goes up in 3D space but down on 2D displays)
				int xProj = int(GRID_SIZE / 2 + K1 * zr * x);
				int yProj = -int(GRID_SIZE / 2 + K1 * zr * y);

				// Luminance (ranges from -root(2) to root(2))
				double lum = cosPhi * cosTheta * sinZrev - cosXrev * cosTheta * sinPhi - sinXrev * sinTheta + cosZrev * (cosXrev * sinTheta - cosTheta * sinXrev * sinPhi);

				// Larger 1/z means the pixel is closer to the viewer than what's already rendered
				if (zr > zBuffer[-yProj][xProj]) {
					zBuffer[-yProj][xProj] = zr;
					// Multiply by 8 to get idx in range 0 - 11 (8 * sqrt(2) = 11.31)
					int lumIdx = int(lum * 8);
					outputGrid[-yProj][xProj] = CHARS[max(lumIdx, 0)];
				}
			}
		}

		renderTorus();
		xrev += 0.04;
		zrev += 0.01;
		sf::sleep(sf::milliseconds(20));
	}
}
