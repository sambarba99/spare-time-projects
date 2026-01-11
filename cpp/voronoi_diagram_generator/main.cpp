/*
Moving Voronoi diagram generator

Controls:
	D: switch distance metric (Euclidean or Manhattan)
	L: activate Lloyd's algorithm for relaxing the Voronoi diagram
	R: randomise seeds
	Up/down arrows: change no. seeds
	Space: play/pause

Author: Sam Barba
Created 11/01/2026
*/

#include <iomanip>
#include <iostream>
#include <random>
#include <SFML/Graphics.hpp>
#include <thread>

using std::vector;


const int IMG_SIZE = 600;
const int MIN_SEEDS = 2;
const int MAX_SEEDS = 50;
const float MAX_VEL_MAGNITUDE = 1.f;
const float POINT_RADIUS = 3.f;
const float LLOYD_TOLERANCE = 0.01f;
const int FPS = 60;

struct Seed {
	float x, y;  // Position
	float vx, vy;  // Velocity
};

int num_seeds = 25;
bool use_euclidean_distance = true;
vector<Seed> seeds;
vector<int> labels(IMG_SIZE * IMG_SIZE);
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> coord_dist(POINT_RADIUS, IMG_SIZE - POINT_RADIUS);
std::uniform_real_distribution<float> vel_dist(-MAX_VEL_MAGNITUDE, MAX_VEL_MAGNITUDE);
sf::RenderWindow window(sf::VideoMode(IMG_SIZE, IMG_SIZE), "Voronoi diagram generator", sf::Style::Close);
sf::Image img;
sf::Texture texture;
sf::Sprite sprite;


void init_seeds() {
	seeds.clear();
	seeds.reserve(num_seeds);
	for (int i = 0; i < num_seeds; i++) {
		seeds.push_back({
			coord_dist(gen),
			coord_dist(gen),
			vel_dist(gen),
			vel_dist(gen)
		});
	}
}


void compute_voronoi(const int y_start, const int y_end) {
	float dx, dy, dist, nearest_dist;

	for (int x = 0; x < IMG_SIZE; x++) {
		for (int y = y_start; y < y_end; y++) {
			nearest_dist = std::numeric_limits<float>::max();
			int best_idx = 0;
			for (int i = 0; i < num_seeds; i++) {
				dx = x - seeds[i].x;
				dy = y - seeds[i].y;
				dist = use_euclidean_distance ? (dx * dx + dy * dy) : (std::abs(dx) + std::abs(dy));
				if (dist < nearest_dist) {
					nearest_dist = dist;
					best_idx = i;
				}
			}
			labels[y * IMG_SIZE + x] = best_idx;
		}
	}
}


void compute_voronoi_threaded() {
	int num_threads = std::thread::hardware_concurrency();  // Use all available cores
	if (num_threads == 0)
		num_threads = 4;  // If detection fails

	std::vector<std::thread> threads;
	int block_height = IMG_SIZE / num_threads;

	for (int i = 0; i < num_threads; i++) {
		int y_start = i * block_height;
		int y_end = (i == num_threads - 1) ? IMG_SIZE : y_start + block_height;  // Last thread takes the remainder

		threads.emplace_back(compute_voronoi, y_start, y_end);
	}

	for (auto& t : threads)
		t.join();
}


bool lloyd_relaxation_step() {
	// https://en.wikipedia.org/wiki/Lloyd%27s_algorithm

	vector<float> sum_x(num_seeds, 0.f);
	vector<float> sum_y(num_seeds, 0.f);
	vector<int> count(num_seeds, 0);

	// Accumulate centroids
	for (int x = 0; x < IMG_SIZE; x++) {
		for (int y = 0; y < IMG_SIZE; y++) {
			int idx = labels[y * IMG_SIZE + x];
			sum_x[idx] += x;
			sum_y[idx] += y;
			count[idx]++;
		}
	}

	float centroid_x, centroid_y, dx, dy;
	float movement, max_movement = 0.f;

	// The more seeds, the larger this step size (0.1 - 0.5) (for equivalent animation speeds)
	float lloyd_step_size = 0.1f + (num_seeds - MIN_SEEDS) * 0.4f / float(MAX_SEEDS - MIN_SEEDS);

	// Move seeds towards centroids
	for (int i = 0; i < num_seeds; i++) {
		if (count[i] == 0)
			continue;

		centroid_x = sum_x[i] / count[i];
		centroid_y = sum_y[i] / count[i];
		dx = centroid_x - seeds[i].x;
		dy = centroid_y - seeds[i].y;

		seeds[i].x += dx * lloyd_step_size;
		seeds[i].y += dy * lloyd_step_size;

		// Keep track of the maximum movement to check convergence
		movement = std::sqrt(dx * dx + dy * dy) * lloyd_step_size;
		if (movement > max_movement)
			max_movement = movement;
	}

	// If true, Lloyd's algorithm has converged
	return max_movement < LLOYD_TOLERANCE;
}


void create_image() {
	img.create(IMG_SIZE, IMG_SIZE, sf::Color(20, 20, 20));

	for (int x = 0; x < IMG_SIZE - 1; x++) {
		for (int y = 0; y < IMG_SIZE - 1; y++) {
			int idx = y * IMG_SIZE + x;
			int idx_down = (y + 1) * IMG_SIZE + x;
			int idx_right = y * IMG_SIZE + x + 1;

			// If when looking at the next pixel down, or next pixel to the right,
			// we see the label is different, then render the current pixel as a border (white)
			if (labels[idx] != labels[idx_down] || labels[idx] != labels[idx_right])
				img.setPixel(x, y, sf::Color::White);
		}
	}
}


void move_points() {
	for (Seed& s : seeds) {
		s.x += s.vx;
		s.y += s.vy;
		if (s.x < POINT_RADIUS + 1 || s.x > IMG_SIZE - POINT_RADIUS)
			s.vx *= -1.f;
		if (s.y < POINT_RADIUS + 1 || s.y > IMG_SIZE - POINT_RADIUS)
			s.vy *= -1.f;
	}
}


void draw() {
	window.clear();

	texture.loadFromImage(img);
	sprite.setTexture(texture);
	window.draw(sprite);

	sf::CircleShape point(POINT_RADIUS);
	point.setFillColor(sf::Color(255, 112, 0));
	point.setOrigin(POINT_RADIUS, POINT_RADIUS);

	for (const Seed& s : seeds) {
		point.setPosition(s.x, s.y);
		window.draw(point);
	}

	window.display();
}


int main() {
	bool lloyd_active = false;
	bool paused = false;
	int screenshot_counter = 0;

	window.setFramerateLimit(FPS);
	sf::Event event;

	init_seeds();

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;

				case sf::Event::KeyPressed:
					switch (event.key.code) {
						case sf::Keyboard::D:
							use_euclidean_distance = !use_euclidean_distance;
							std::cout << "Distance metric: " << (use_euclidean_distance ? "Euclidean\n" : "Manhattan\n");
							break;

						case sf::Keyboard::L:
							lloyd_active = true;
							paused = false;
							std::cout << "Running Lloyd's algorithm\n";
							break;

						case sf::Keyboard::R:
							init_seeds();
							lloyd_active = paused = false;
							break;

						case sf::Keyboard::Up:
							if (num_seeds < MAX_SEEDS) {
								num_seeds++;
								init_seeds();
								lloyd_active = paused = false;
								std::cout << "No. seeds: " << num_seeds << '\n';
							}
							break;

						case sf::Keyboard::Down:
							if (num_seeds > MIN_SEEDS) {
								num_seeds--;
								init_seeds();
								lloyd_active = paused = false;
								std::cout << "No. seeds: " << num_seeds << '\n';
							}
							break;

						case sf::Keyboard::Space:
							paused = !paused;
							break;
					}
					break;
			}
		}

		if (paused)
			continue;

		compute_voronoi_threaded();
		if (lloyd_active) {
			if (lloyd_relaxation_step()) {
				lloyd_active = false;
				paused = true;
				std::cout << "Converged\n";
			}
		} else {
			move_points();
		}
		create_image();
		draw();

		// sf::Texture screenshot_texture;
		// sf::Image screenshot;
		// screenshot_texture.create(window.getSize().x, window.getSize().y);
		// screenshot_texture.update(window);
		// screenshot = screenshot_texture.copyToImage();
		// std::ostringstream file_path;
		// file_path << "C:/Users/sam/Desktop/frames/" << std::setw(4) << std::setfill('0') << screenshot_counter << ".png";
		// screenshot.saveToFile(file_path.str());
		// screenshot_counter++;
	}

	return 0;
}
