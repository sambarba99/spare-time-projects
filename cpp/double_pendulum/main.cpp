/*
Double Pendulum simulator

Controls:
	R: reset
	Space: play/pause

Author: Sam Barba
Created 15/11/2022
*/

#include <cmath>
#include <iomanip>
#include <random>
#include <SFML/Graphics.hpp>


const double R1 = 250;
const double R2 = 250;
const double M1 = 10;
const double M2 = 10;
const double G = 0.3;
const double DAMPING_FACTOR = 0.9999;
const double COLOUR_DECAY = 0.998;
const int WIDTH = 1000;
const int HEIGHT = 600;
const int FPS = 60;

double a1, a2, vel1, vel2;
std::vector<std::pair<int, int>> positions;
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dist(0, 2 * M_PI);
sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Double Pendulum", sf::Style::Close);
int screenshotCounter = 0;


void drawLine(int x1, int y1, const int x2, const int y2, const int red) {
	// Bresenham's algorithm

	int dx = abs(x1 - x2);
	int dy = -abs(y1 - y2);
	int sx = x1 < x2 ? 1 : -1;
	int sy = y1 < y2 ? 1 : -1;
	int err = dx + dy;
	int e2;

	while (true) {
		sf::Vertex pix(sf::Vector2f(x1, y1), sf::Color(red, 0, 0));
		window.draw(&pix, 1, sf::Points);

		if (x1 == x2 && y1 == y2) return;

		e2 = 2 * err;
		if (e2 >= dy) {
			err += dy;
			x1 += sx;
		}
		if (e2 <= dx) {
			err += dx;
			y1 += sy;
		}
	}
}


void draw() {
	window.clear(sf::Color::Black);

	double num1 = -G * (2 * M1 + M2) * sin(a1);
	double num2 = -M2 * G * sin(a1 - 2 * a2);
	double num3 = -2 * sin(a1 - a2) * M2;
	double num4 = vel2 * vel2 * R2 + vel1 * vel1 * R1 * cos(a1 - a2);
	double den = R1 * (2 * M1 + M2 - M2 * cos(2 * a1 - 2 * a2));
	double a1acc = (num1 + num2 + num3 * num4) / den;

	num1 = 2 * sin(a1 - a2);
	num2 = vel1 * vel1 * R1 * (M1 + M2);
	num3 = G * (M1 + M2) * cos(a1);
	num4 = vel2 * vel2 * R2 * M2 * cos(a1 - a2);
	den = R2 * (2 * M1 + M2 - M2 * cos(2 * a1 - 2 * a2));
	double a2acc = num1 * (num2 + num3 + num4) / den;

	double x1 = R1 * sin(a1) + WIDTH / 2.0;
	double y1 = R1 * cos(a1);
	double x2 = x1 + R2 * sin(a2);
	double y2 = y1 + R2 * cos(a2);

	sf::Vertex line1[] = {
		sf::Vertex(sf::Vector2f(WIDTH / 2, 0), sf::Color(220, 220, 220)),
		sf::Vertex(sf::Vector2f(x1, y1), sf::Color(220, 220, 220))
	};
	sf::Vertex line2[] = {
		sf::Vertex(sf::Vector2f(x1, y1), sf::Color(220, 220, 220)),
		sf::Vertex(sf::Vector2f(x2, y2), sf::Color(220, 220, 220))
	};
	sf::CircleShape circle1(10.f);
	sf::CircleShape circle2(10.f);
	circle1.setPosition(x1 - 10, y1 - 10);
	circle2.setPosition(x2 - 10, y2 - 10);
	circle1.setFillColor(sf::Color(220, 220, 220));
	circle2.setFillColor(sf::Color(220, 220, 220));
	window.draw(line1, 2, sf::Lines);
	window.draw(line2, 2, sf::Lines);
	window.draw(circle1);
	window.draw(circle2);

	positions.push_back({int(x2), int(y2)});
	if (positions.size() > 1) {
		int n = positions.size();
		for (int i = 0; i < n - 1; i++) {
			double red = 255 * pow(COLOUR_DECAY, n - i);
			drawLine(positions[i].first, positions[i].second, positions[i + 1].first, positions[i + 1].second, int(red));
		}
	}

	if (positions.size() > 1000) positions.erase(positions.begin());

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

	vel1 += a1acc;
	vel2 += a2acc;
	a1 += vel1;
	a2 += vel2;

	vel1 *= DAMPING_FACTOR;
	vel2 *= DAMPING_FACTOR;
}


int main() {
	window.setFramerateLimit(FPS);

	a1 = dist(gen);
	a2 = dist(gen);
	vel1 = vel2 = 0.0;
	bool paused = false;
	sf::Event event;

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::KeyPressed:
					switch (event.key.code) {
						case sf::Keyboard::R:
							a1 = dist(gen);
							a2 = dist(gen);
							vel1 = vel2 = 0.0;
							positions.clear();
							paused = false;
							break;
						case sf::Keyboard::Space:
							paused = !paused;
							break;
					}
					break;
			}
		}

		if (paused) continue;

		draw();
	}

	return 0;
}
