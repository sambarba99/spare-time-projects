/*
Perfectly elastic collision simulator

Author: Sam Barba
Created 28/08/2024
*/

#include <cmath>
#include <iomanip>
#include <random>
#include <SFML/Graphics.hpp>

using std::vector;


const float MIN_MASS = 100.f;
const float MAX_MASS = 1500.f;
const float MAX_VEL_MAGNITUDE = 3.f;
const int NUM_PARTICLES = 100;
const int WIDTH = 1200;
const int HEIGHT = 750;
const int FPS = 60;

class Particle {
	public:
		float mass;
		float radius;
		sf::Vector2f pos;
        sf::Vector2f vel;
        int hue;

		Particle(const float mass, const float radius, const sf::Vector2f& pos, const sf::Vector2f& vel) {
			this->mass = mass;
            this->radius = sqrt(mass);
            this->pos = pos;
            this->vel = vel;
            this->hue = (1 - (mass - MIN_MASS) / (MAX_MASS - MIN_MASS)) * 60;
		}

		void update() {
			pos += vel;
			if (pos.x < radius) {
				vel.x *= -1;
				pos.x = radius;
			} else if (pos.x > WIDTH - radius) {
				vel.x *= -1;
				pos.x = WIDTH - radius;
			}
			if (pos.y < radius) {
				vel.y *= -1;
				pos.y = radius;
			} else if (pos.y > HEIGHT - radius) {
				vel.y *= -1;
				pos.y = HEIGHT - radius;
			}
		}

		void collide(Particle& other) {
			sf::Vector2f deltaPos = other.pos - pos;
			float dist = std::sqrt(deltaPos.x * deltaPos.x + deltaPos.y * deltaPos.y);

			if (dist > radius + other.radius) return;

			sf::Vector2f impactVector = other.pos - pos;

			// Push particles apart so they aren't overlapping
			float overlap = dist - (radius + other.radius);
			dist += 1e-6;  // Avoid division by 0
			float factor = overlap * 0.5f / dist;  // Vector multiplication factor = (desired length) / (current length)
			deltaPos *= factor;
			pos += deltaPos;
			other.pos -= deltaPos;

			// Correct the distance
			dist = radius + other.radius;
			float currentLength = std::sqrt(impactVector.x * impactVector.x + impactVector.y * impactVector.y);
			factor = dist / currentLength;
			impactVector *= factor;

			// Numerators for updating this particle (A) and other particle (B), and denominator for both
			sf::Vector2f relativeVel = other.vel - vel;
			float dotProd = relativeVel.x * impactVector.x + relativeVel.y * impactVector.y;
			float numA = dotProd * 2 * other.mass;
			float numB = dotProd * -2 * mass;
			float den = (mass + other.mass) * dist * dist;

			// Update this particle (A)
			sf::Vector2f deltaVelA = impactVector * numA / den;
			vel += deltaVelA;

			// Update other particle (B)
			sf::Vector2f deltaVelB = impactVector * numB / den;
			other.vel += deltaVelB;
		}
};

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> massDist(MIN_MASS, MAX_MASS);
std::uniform_real_distribution<float> velDist(-MAX_VEL_MAGNITUDE, MAX_VEL_MAGNITUDE);
vector<Particle> particles;
sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Elastic Collisions", sf::Style::Close);
int screenshotCounter = 0;


vector<int> hsv2rgb(const float h, const float s, const float v) {
	/*
	HSV to RGB
	0 <= hue < 360
	0 <= saturation <= 1
	0 <= value <= 1
	*/

	float c = s * v;
	float x = c * (1 - fabs(fmod(h / 60.0, 2) - 1));
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

	return {r, g, b};
}


void draw() {
	window.clear(sf::Color::Black);

	for (const Particle& p : particles) {
		sf::CircleShape circle(p.radius);
		circle.setPosition(p.pos.x - p.radius, p.pos.y - p.radius);
		vector<int> rgb = hsv2rgb(float(p.hue), 1.f, 1.f);
		circle.setFillColor(sf::Color(rgb[0], rgb[1], rgb[2]));
		window.draw(circle);
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

	for (int i = 0; i < NUM_PARTICLES; i++) {
		float mass = massDist(gen);
		float radius = sqrt(mass);
		std::uniform_real_distribution<float> xPosDist(radius, WIDTH - radius);
		std::uniform_real_distribution<float> yPosDist(radius, HEIGHT - radius);
		sf::Vector2f pos(xPosDist(gen), yPosDist(gen));
		sf::Vector2f vel(velDist(gen), velDist(gen));

		particles.push_back(Particle(mass, radius, pos, vel));
	}

	sf::Event event;

	while (window.isOpen()) {
		while (window.pollEvent(event))
			if (event.type == sf::Event::Closed)
				window.close();

		for (Particle& p : particles)
			p.update();

		for (int i = 0; i < NUM_PARTICLES - 1; i++)
			for (int j = i + 1; j < NUM_PARTICLES; j++)
				particles[i].collide(particles[j]);

		draw();
	}

	return 0;
}
