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
			// Damping
			// vel.x *= 0.999;
			// vel.y *= 0.999;
		}

		void collide(Particle& other) {
			sf::Vector2f delta_pos = other.pos - pos;
			float dist = std::sqrt(delta_pos.x * delta_pos.x + delta_pos.y * delta_pos.y);

			if (dist > radius + other.radius)
				return;

			sf::Vector2f impact_vector = other.pos - pos;

			// Push particles apart so they aren't overlapping
			float overlap = dist - (radius + other.radius);
			dist += 1e-6;  // Avoid division by 0
			float factor = overlap * 0.5f / dist;  // Vector multiplication factor = (desired length) / (current length)
			delta_pos *= factor;
			pos += delta_pos;
			other.pos -= delta_pos;

			// Correct the distance
			dist = radius + other.radius;
			float current_length = std::sqrt(impact_vector.x * impact_vector.x + impact_vector.y * impact_vector.y);
			factor = dist / current_length;
			impact_vector *= factor;

			// Numerators for updating this particle (A) and other particle (B), and denominator for both
			sf::Vector2f relative_vel = other.vel - vel;
			float dot_prod = relative_vel.x * impact_vector.x + relative_vel.y * impact_vector.y;
			float numA = dot_prod * 2 * other.mass;
			float numB = dot_prod * -2 * mass;
			float den = (mass + other.mass) * dist * dist;

			// Update this particle (A)
			sf::Vector2f delta_vel_a = impact_vector * numA / den;
			vel += delta_vel_a;

			// Update other particle (B)
			sf::Vector2f delta_vel_b = impact_vector * numB / den;
			other.vel += delta_vel_b;
		}
};

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> mass_dist(MIN_MASS, MAX_MASS);
std::uniform_real_distribution<float> vel_dist(-MAX_VEL_MAGNITUDE, MAX_VEL_MAGNITUDE);
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
	window.clear();

	for (const Particle& p : particles) {
		sf::CircleShape circle(p.radius);
		circle.setPosition(p.pos.x - p.radius, p.pos.y - p.radius);
		vector<int> rgb = hsv2rgb(float(p.hue), 1.f, 1.f);
		circle.setFillColor(sf::Color(rgb[0], rgb[1], rgb[2]));
		window.draw(circle);
	}

	window.display();
}


int main() {
	particles.reserve(NUM_PARTICLES);

	int screenshot_counter = 0;
	window.setFramerateLimit(FPS);

	for (int i = 0; i < NUM_PARTICLES; i++) {
		float mass = mass_dist(gen);
		float radius = sqrt(mass);
		std::uniform_real_distribution<float> x_pos_dist(radius, WIDTH - radius);
		std::uniform_real_distribution<float> y_pos_dist(radius, HEIGHT - radius);
		sf::Vector2f pos(x_pos_dist(gen), y_pos_dist(gen));
		sf::Vector2f vel(vel_dist(gen), vel_dist(gen));

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
		// sf::Texture texture;
		// sf::Image screenshot;
		// texture.create(window.getSize().x, window.getSize().y);
		// texture.update(window);
		// screenshot = texture.copyToImage();
		// std::ostringstream filePath;
		// filePath << "C:/Users/sam/Desktop/frames/" << std::setw(4) << std::setfill('0') << screenshot_counter << ".png";
		// screenshot.saveToFile(filePath.str());
		// screenshot_counter++;
	}

	return 0;
}
