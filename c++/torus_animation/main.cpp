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

char output_grid[GRID_SIZE][GRID_SIZE];
sf::RenderWindow window(sf::VideoMode(GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE), "Revolving torus", sf::Style::Close);
sf::Font font;


void draw() {
	window.clear();

	for (int y = 0; y < GRID_SIZE; y++) {
		for (int x = 0; x < GRID_SIZE; x++) {
			sf::Text text(output_grid[y][x], font, 14);
			text.setPosition(int(x * CELL_SIZE), int(y * CELL_SIZE));
			text.setFillColor(sf::Color::White);
			window.draw(text);
		}
	}

	window.display();
}


int main() {
	// Revolution amounts about x and z axes (increased after each frame, to revolve the torus)
	double xrev, zrev;
	double sin_xrev, cos_xrev, sin_zrev, cos_zrev;
	double sin_theta, cos_theta, sin_phi, cos_phi;
	double z_buffer[GRID_SIZE][GRID_SIZE];
	bool paused = false;
	int screenshot_counter = 0;
	sf::Event event;

	window.setFramerateLimit(FPS);
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");

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

		if (paused)
			continue;

		sin_xrev = sin(xrev);
		cos_xrev = cos(xrev);
		sin_zrev = sin(zrev);
		cos_zrev = cos(zrev);

		std::fill(&output_grid[0][0], &output_grid[0][0] + sizeof(output_grid) / sizeof(output_grid[0][0]), ' ');
		std::fill(&z_buffer[0][0], &z_buffer[0][0] + sizeof(z_buffer) / sizeof(z_buffer[0][0]), 0.0);

		for (double theta = 0.0; theta < 2 * M_PI; theta += CHANGE_THETA) {
			sin_theta = sin(theta);
			cos_theta = cos(theta);

			for (double phi = 0.0; phi < 2 * M_PI; phi += CHANGE_PHI) {
				sin_phi = sin(phi);
				cos_phi = cos(phi);

				// x,y coords before revolution
				double circle_x = R2 + R1 * cos_theta;
				double circle_y = R1 * sin_theta;

				// 3D coords after revolution
				double x = circle_x * (cos_zrev * cos_phi + sin_xrev * sin_zrev * sin_phi) - circle_y * cos_xrev * sin_zrev;
				double y = circle_x * (sin_zrev * cos_phi - sin_xrev * cos_zrev * sin_phi) + circle_y * cos_xrev * cos_zrev;
				double z = K1 + cos_xrev * circle_x * sin_phi + circle_y * sin_xrev;
				double zr = 1.0 / z;

				// x, y projection (y is negated, as y goes up in 3D space but down on 2D displays)
				int x_proj = int(GRID_SIZE / 2 + K2 * zr * x);
				int y_proj = -int(GRID_SIZE / 2 + K2 * zr * y);

				// Luminance (ranges from -sqrt(2) to sqrt(2))
				double lum = cos_phi * cos_theta * sin_zrev - cos_xrev * cos_theta * sin_phi - sin_xrev * sin_theta + cos_zrev * (cos_xrev * sin_theta - cos_theta * sin_xrev * sin_phi);

				// Larger 1/z means the pixel is closer to the viewer than what's already rendered
				if (zr > z_buffer[-y_proj][x_proj]) {
					z_buffer[-y_proj][x_proj] = zr;
					// Multiply by 8 to get idx in range 0 - 11 (8 * sqrt(2) = 11.31)
					int lum_idx = lum * 8;
					output_grid[-y_proj][x_proj] = CHARS[std::max(lum_idx, 0)];
				}
			}
		}

		draw();
		// sf::Texture texture;
		// sf::Image screenshot;
		// texture.create(window.getSize().x, window.getSize().y);
		// texture.update(window);
		// screenshot = texture.copyToImage();
		// std::ostringstream file_path;
		// file_path << "C:/Users/sam/Desktop/frames/" << std::setw(4) << std::setfill('0') << screenshot_counter << ".png";
		// screenshot.saveToFile(file_path.str());
		// screenshot_counter++;

		xrev += CHANGE_X_REV;
		zrev += CHANGE_Z_REV;
	}

	return 0;
}
