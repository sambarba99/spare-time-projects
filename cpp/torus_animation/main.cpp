/*
Revolving torus animation (press space to toggle animation)

Author: Sam Barba
Created 15/11/2022
*/

#include <cmath>
#include <iomanip>
#include <SFML/Graphics.hpp>


const int GRID_SIZE = 60;  // Rows and cols
const int CELL_SIZE = 11;
const int FPS = 30;

const double R1 = 2.0;  // Radius of torus cross-sectional circle
const double R2 = 5.0;  // Distance from origin to centre of cross-sectional circle

const double CHANGE_THETA = 2 * M_PI / 100.0;  // Theta goes around the cross-sectional circle of the torus
const double CHANGE_PHI = 2 * M_PI / 200.0;  // Phi revolves this circle around the y-axis, creating the torus (solid of revolution)

// How much to revolve the torus (about x and z axes) per frame
const double CHANGE_X_REV = M_PI * 0.014;
const double CHANGE_Z_REV = CHANGE_X_REV / 4;

/*
Calculate K2 based on screen size: the maximum x-distance occurs roughly at the edge of the torus,
which is at x = R1 + R2, z = 0. We want that to be displaced 3/8ths of the width of the screen.
*/
const double K1 = 100.0;  // Arbitrary distance from torus to viewer
const double K2 = GRID_SIZE * K1 * 3 / (8 * (R1 + R2));
const std::string CHARS = "@$#=*!;:~-,.";  // 'Brightest' to 'dimmest' chars

char outputGrid[GRID_SIZE][GRID_SIZE];
sf::RenderWindow window(sf::VideoMode(GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE), "Revolving torus", sf::Style::Close);
sf::Font font;
int screenshotCounter = 0;


void renderTorus() {
	window.clear(sf::Color::Black);

	for (int y = 0; y < GRID_SIZE; y++) {
		for (int x = 0; x < GRID_SIZE; x++) {
			sf::Text text(outputGrid[y][x], font, 14);
			text.setPosition(int(x * CELL_SIZE), int(y * CELL_SIZE));
			text.setFillColor(sf::Color::White);
			window.draw(text);
		}
	}

	window.display();

	// sf::Texture texture;
	// sf::Image screenshot;
	// texture.create(window.getSize().x, window.getSize().y);
	// texture.update(window);
	// screenshot = texture.copyToImage();
	// std::ostringstream filePath;
	// filePath << "C:/Users/sam/Desktop/frames/" << std::setw(4) << std::setfill('0') << screenshotCounter << ".png";
	// screenshot.saveToFile(filePath.str());
	// screenshotCounter++;
}


int main() {
	window.setFramerateLimit(FPS);
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");

	// Revolution amounts about x and z axes (increased after each frame, to revolve the torus)
	double xrev, zrev;
	double sinXrev, cosXrev, sinZrev, cosZrev;
	double sinTheta, cosTheta, sinPhi, cosPhi;
	double zBuffer[GRID_SIZE][GRID_SIZE];
	bool paused = false;
	sf::Event event;

	while (window.isOpen()) {
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

		std::fill(&outputGrid[0][0], &outputGrid[0][0] + sizeof(outputGrid) / sizeof(outputGrid[0][0]), ' ');
		std::fill(&zBuffer[0][0], &zBuffer[0][0] + sizeof(zBuffer) / sizeof(zBuffer[0][0]), 0.0);

		for (double theta = 0.0; theta < 2 * M_PI; theta += CHANGE_THETA) {
			sinTheta = sin(theta);
			cosTheta = cos(theta);

			for (double phi = 0.0; phi < 2 * M_PI; phi += CHANGE_PHI) {
				sinPhi = sin(phi);
				cosPhi = cos(phi);

				// x,y coords before revolution
				double circleX = R2 + R1 * cosTheta;
				double circleY = R1 * sinTheta;

				// 3D coords after revolution
				double x = circleX * (cosZrev * cosPhi + sinXrev * sinZrev * sinPhi) - circleY * cosXrev * sinZrev;
				double y = circleX * (sinZrev * cosPhi - sinXrev * cosZrev * sinPhi) + circleY * cosXrev * cosZrev;
				double z = K1 + cosXrev * circleX * sinPhi + circleY * sinXrev;
				double zr = 1.0 / z;

				// x, y projection (y is negated, as y goes up in 3D space but down on 2D displays)
				int xProj = int(GRID_SIZE / 2 + K2 * zr * x);
				int yProj = -int(GRID_SIZE / 2 + K2 * zr * y);

				// Luminance (ranges from -sqrt(2) to sqrt(2))
				double lum = cosPhi * cosTheta * sinZrev - cosXrev * cosTheta * sinPhi - sinXrev * sinTheta + cosZrev * (cosXrev * sinTheta - cosTheta * sinXrev * sinPhi);

				// Larger 1/z means the pixel is closer to the viewer than what's already rendered
				if (zr > zBuffer[-yProj][xProj]) {
					zBuffer[-yProj][xProj] = zr;
					// Multiply by 8 to get idx in range 0 - 11 (8 * sqrt(2) = 11.31)
					int lumIdx = lum * 8;
					outputGrid[-yProj][xProj] = CHARS[std::max(lumIdx, 0)];
				}
			}
		}

		renderTorus();
		xrev += CHANGE_X_REV;
		zrev += CHANGE_Z_REV;
	}

	return 0;
}
