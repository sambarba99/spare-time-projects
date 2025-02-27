/*
Perfectly elastic collision simulator

Author: Sam Barba
Created 28/08/2024
*/

#include <iostream>
#include <cmath>
#include <random>
#include <SFML/Graphics.hpp>

#include "particle.h"

using std::cout;
using std::vector;


const double MIN_MASS = 100.f;
const double MAX_MASS = 1500.f;
const double MAX_VEL_MAGNITUDE = 3.f;
const int NUM_PARTICLES = 10;
const int WIDTH = 1200;
const int HEIGHT = 800;
const int FPS = 60;

std::random_device rd;
std::mt19937 gen(rd());
vector<Particle> particles;
sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Elastic Collisions", sf::Style::Close);


vector<int> hsv2rgb(const float h, const float s, const float v) {
	/*
	HSV to RGB
	0 <= hue < 360
	0 <= saturation <= 1
	0 <= value <= 1
	*/

	float c = s * v;
	float x = c * (1 - abs(fmod(h / 60.0, 2) - 1));
	float m = v - c;

	float rf, gf, bf;
	if (h < 60) rf = c, gf = x, bf = 0;
	else if (h < 120) rf = x, gf = c, bf = 0;
	else if (h < 180) rf = 0, gf = c, bf = x;
	else if (h < 240) rf = 0, gf = x, bf = c;
	else if (h < 300) rf = x, gf = 0, bf = c;
	else rf = c, gf = 0, bf = x;

	int r = (rf + m) * 255;
	int g = (gf + m) * 255;
	int b = (bf + m) * 255;

	cout << h << ' ' << s << ' ' << v << '\n';
	cout << c << ' ' << x << ' ' << m << '\n';
	cout << r << ' ' << g << ' ' << b << '\n' << '\n';

	return {r, g, b};
}


void draw() {
	window.clear(sf::Color::Black);

	for (Particle p : particles) {
		sf::CircleShape circle(p.radius);
		circle.setPosition(p.pos.x, p.pos.y);
		vector<int> rgb = hsv2rgb(float(p.hue), 1.f, 1.f);
		circle.setFillColor(sf::Color(rgb[0], rgb[1], rgb[2]));
		window.draw(circle);
	}

	window.display();
}


int main() {
	window.setFramerateLimit(FPS);

	for (int i = 0; i < NUM_PARTICLES; i++) {
		std::uniform_real_distribution<double> massDist(MIN_MASS, MAX_MASS);
		double mass = massDist(gen);
		double radius = sqrt(mass);
		std::uniform_real_distribution<double> xPosDist(radius, WIDTH - radius);
		std::uniform_real_distribution<double> yPosDist(radius, HEIGHT - radius);
		sf::Vector2f pos(xPosDist(gen), yPosDist(gen));
		std::uniform_real_distribution<double> velDist(-MAX_VEL_MAGNITUDE, MAX_VEL_MAGNITUDE);
		sf::Vector2f vel(velDist(gen), velDist(gen));
		int hue = (mass - MIN_MASS) / (MAX_MASS - MIN_MASS) * 60;

		particles.push_back(Particle(mass, radius, pos, vel, hue));
	}

	sf::Event event;

	while (window.isOpen()) {
		while (window.pollEvent(event))
			if (event.type == sf::Event::Closed)
				window.close();

		for (Particle p : particles)
			p.update();

		for (int i = 0; i < NUM_PARTICLES - 1; i++)
			for (int j = i + 1; j < NUM_PARTICLES; j++)
				particles[i].collide(particles[j]);

		draw();
	}
}
