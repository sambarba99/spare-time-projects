/*
Ray casting demo

Controls:
	WASD: move around
	R: reset

Author: Sam Barba
Created 15/11/2022
*/

#include <optional>
#include <random>
#include <SFML/Graphics.hpp>

using std::vector;


// World constants
const int SIZE_PX = 800;  // Size of world (px)
const int SIZE_GRID = 10;  // Size of grid world (squares)
const int GRID_SQUARE_SIZE = SIZE_PX / SIZE_GRID;

// Ray/rendering constants
const int NUM_RAYS = SIZE_PX;
const double MAX_RAY_LENGTH = std::sqrt(2 * SIZE_PX * SIZE_PX);
const double FOV_ANGLE = M_PI / 3.0;  // Field of view angle = 60 deg
const double DELTA_ANGLE = FOV_ANGLE / NUM_RAYS;
const double SCREEN_DIST = SIZE_PX * 0.5 / tan(FOV_ANGLE * 0.5);
const int WALL_WIDTH = SIZE_PX / NUM_RAYS;  // Width (px) of each wall segment to render in 3D
const double PROJ_HEIGHT_SCALE = 75.0;  // Tweak until the rendered 3D boxes look approx. cubic (otherwise can look squished/stretched)
const int FPS = 60;

// Player constants
const double MOVEMENT_SPEED = 3.5;
const double TURNING_SPEED = 0.8;
const int POINT_RADIUS = 5;

struct Line {
	double x1;
	double y1;
	double x2;
	double y2;
	// These 2 attributes are used when the line represents a projected ray (instead of a wall obstacle)
	double length;
	double proj_height;
};

struct Circle {
	double x;
	double y;
	double r;
};


vector<Line> rays;
vector<Line> walls;
vector<Circle> circles;
double player_x, player_y, player_heading;
std::random_device rd;
std::mt19937 gen(rd());
sf::RenderWindow window(sf::VideoMode(SIZE_PX * 2, SIZE_PX), "Ray casting demo", sf::Style::Close);
sf::Font font;


vector<Line> make_box(const int grid_idx) {
	double x1 = (grid_idx % SIZE_GRID) * GRID_SQUARE_SIZE;
	double y1 = (grid_idx / SIZE_GRID) * GRID_SQUARE_SIZE;
	double x2 = x1 + GRID_SQUARE_SIZE;
	double y2 = y1 + GRID_SQUARE_SIZE;
	Line north_wall = {x1, y1, x2, y1};
	Line east_wall = {x2, y1, x2, y2};
	Line south_wall = {x2, y2, x1, y2};
	Line west_wall = {x1, y2, x1, y1};
	return {north_wall, east_wall, south_wall, west_wall};
}


void generate_obstacles(const int num_boxes = 5, const int num_circles = 5) {
	// Shuffle all grid indices for sampling
	vector<int> all_grid_indices(SIZE_GRID * SIZE_GRID);
	std::iota(all_grid_indices.begin(), all_grid_indices.end(), 0);
	std::shuffle(all_grid_indices.begin(), all_grid_indices.end(), gen);

	walls.clear();
	for (int i = 0; i < num_boxes; i++) {
		vector<Line> box_walls = make_box(all_grid_indices[i]);
		for (const Line& wall : box_walls)
			walls.emplace_back(wall);
	}

	circles.clear();
	for (int i = num_boxes; i < num_boxes + num_circles; i++) {
		int grid_idx = all_grid_indices[i];
		double cx = (grid_idx % SIZE_GRID) * GRID_SQUARE_SIZE + GRID_SQUARE_SIZE / 2;
		double cy = (grid_idx / SIZE_GRID) * GRID_SQUARE_SIZE + GRID_SQUARE_SIZE / 2;
		circles.push_back({cx, cy, GRID_SQUARE_SIZE / 2});
	}

	// World border
	Line north_border = {0, 0, SIZE_PX, 0};
	Line east_border = {SIZE_PX, 0, SIZE_PX, SIZE_PX};
	Line south_border = {0, SIZE_PX - 1, SIZE_PX, SIZE_PX - 1};
	Line west_border = {1, 0, 1, SIZE_PX};
	walls.insert(walls.end(), {north_border, east_border, south_border, west_border});
}


std::optional<vector<double>> find_ray_wall_intersection(const Line& ray, const Line& wall) {
	double rx1 = ray.x1, ry1 = ray.y1;
	double rx2 = ray.x2, ry2 = ray.y2;
	double wx1 = wall.x1, wy1 = wall.y1;
	double wx2 = wall.x2, wy2 = wall.y2;

	double denom = (rx1 - rx2) * (wy1 - wy2) - (ry1 - ry2) * (wx1 - wx2);
	if (denom == 0)
		return std::nullopt;

	double t = ((rx1 - wx1) * (wy1 - wy2) - (ry1 - wy1) * (wx1 - wx2)) / denom;
	double u = -((rx1 - rx2) * (ry1 - wy1) - (ry1 - ry2) * (rx1 - wx1)) / denom;

	if (0 <= t && t <= 1 && 0 <= u && u <= 1) {
		double intersection_x = rx1 + t * (rx2 - rx1);
		double intersection_y = ry1 + t * (ry2 - ry1);
		return vector<double>{intersection_x, intersection_y};
	}

	return std::nullopt;
}


std::optional<vector<double>> find_ray_circle_intersection(const Line& ray, const Circle& circle) {
	double rx1 = ray.x1, ry1 = ray.y1;
	double rx2 = ray.x2, ry2 = ray.y2;
	double cx = circle.x, cy = circle.y;
	double r = circle.r;

	double dx = rx2 - rx1;
	double dy = ry2 - ry1;
	double mx = rx1 - cx;
	double my = ry1 - cy;

	double a = dx * dx + dy * dy;
	double b = 2 * (mx * dx + my * dy);
	double c = mx * mx + my * my - r * r;
	double discriminant = b * b - 4 * a * c;

	if (discriminant < 0)
		return std::nullopt;

	double sqrt_d = std::sqrt(discriminant);
	double t1 = (-b - sqrt_d) / (2 * a);
	double t2 = (-b + sqrt_d) / (2 * a);

	vector<double> valid_ts;
	if (0 <= t1 && t1 <= 1)
		valid_ts.emplace_back(t1);
	if (0 <= t2 && t2 <= 1)
		valid_ts.emplace_back(t2);
	if (valid_ts.empty())
		return std::nullopt;

	double t = *std::min_element(valid_ts.begin(), valid_ts.end());
	double intersection_x = rx1 + t * dx;
	double intersection_y = ry1 + t * dy;

	return vector<double>{intersection_x, intersection_y};
}


void generate_rays() {
	rays.clear();
	double ray_angle = player_heading - FOV_ANGLE / 2.0;
	double end_x, end_y, ix, iy, length, corrected_dist, proj_height;

	for (int i = 0; i < NUM_RAYS; i++) {
		// Initialise ray arbitrarily long (2 * SIZE_PX ensures it goes off the screen).
		// After the following loops, ray.length will be the distance to the nearest object that the ray hits.
		end_x = 2 * SIZE_PX * cos(ray_angle) + player_x;
		end_y = 2 * SIZE_PX * sin(ray_angle) + player_y;
		Line ray = {player_x, player_y, end_x, end_y};

		for (const Line& wall : walls) {
			std::optional<vector<double>> intersection = find_ray_wall_intersection(ray, wall);
			if (intersection) {
				ray.x2 = (*intersection)[0];
				ray.y2 = (*intersection)[1];
			}
		}

		for (const Circle& circle : circles) {
			std::optional<vector<double>> intersection = find_ray_circle_intersection(ray, circle);
			if (intersection) {
				ray.x2 = (*intersection)[0];
				ray.y2 = (*intersection)[1];
			}
		}

		ray.length = std::sqrt(std::pow(ray.x1 - ray.x2, 2) + std::pow(ray.y1 - ray.y2, 2));
		corrected_dist = ray.length * cos(player_heading - ray_angle);  // Remove fisheye distortion
		corrected_dist += 1e-6;  // Avoid division by 0
		proj_height = SCREEN_DIST / corrected_dist * PROJ_HEIGHT_SCALE;
		ray.proj_height = std::min(proj_height, double(SIZE_PX));

		rays.emplace_back(ray);
		ray_angle += DELTA_ANGLE;
	}
}


double map_range(const double x, const double from_lo, const double from_hi, const double to_lo, const double to_hi) {
	// Map x from [from_lo, from_hi] to [to_lo, to_hi]
	if (from_hi - from_lo == 0)
		return to_hi;
	return (x - from_lo) / (from_hi - from_lo) * (to_hi - to_lo) + to_lo;
}


void draw() {
	window.clear();

	// ---------- Left half of screen (bird's-eye view) ----------

	// Ground
	sf::RectangleShape ground_2d(sf::Vector2f(SIZE_PX, SIZE_PX));
	ground_2d.setPosition(0, 0);
	ground_2d.setFillColor(sf::Color(160, 160, 160));
	window.draw(ground_2d);

	// 2D rays
	for (int i = 0; i < rays.size(); i++) {
		if (i % 20 == 0) {
			Line ray = rays[i];
			sf::Vertex line[] = {
				sf::Vertex(sf::Vector2f(ray.x1, ray.y1), sf::Color::Red),
				sf::Vertex(sf::Vector2f(ray.x2, ray.y2), sf::Color::Red)
			};
			window.draw(line, 2, sf::Lines);
		}
	}

	// Player
	sf::CircleShape player_circle(POINT_RADIUS);
	player_circle.setPosition(player_x - POINT_RADIUS, player_y - POINT_RADIUS);
	player_circle.setFillColor(sf::Color::Black);
	window.draw(player_circle);

	// 2D boxes
	for (int i = 0; i + 3 < walls.size() - 4; i += 4) {
		// North-west, north-east, south-east, south-west corners
		vector<sf::Vector2f> corners = {
			sf::Vector2f(walls[i].x1, walls[i].y1),
			sf::Vector2f(walls[i + 1].x1, walls[i + 1].y1),
			sf::Vector2f(walls[i + 2].x1, walls[i + 2].y1),
			sf::Vector2f(walls[i + 3].x1, walls[i + 3].y1)
		};
		sf::ConvexShape box(4);
		for (int j = 0; j < 4; j++)
			box.setPoint(j, corners[j]);
		box.setFillColor(sf::Color::White);
		window.draw(box);
	}

	// 2D circles
	for (const Circle& c : circles) {
		sf::CircleShape circle(c.r);
		circle.setPosition(c.x - c.r, c.y - c.r);
		circle.setFillColor(sf::Color::White);
		window.draw(circle);
	}

	// World border (last 4 walls)
	for (int i = walls.size() - 4; i < walls.size(); i++) {
		sf::Vertex line[] = {
			sf::Vertex(sf::Vector2f(walls[i].x1, walls[i].y1), sf::Color::Black),
			sf::Vertex(sf::Vector2f(walls[i].x2, walls[i].y2), sf::Color::Black)
		};
		window.draw(line, 2, sf::Lines);
	}

	// ---------- Right half of screen (3D POV) ----------

	// Sky and ground
	sf::RectangleShape sky(sf::Vector2f(SIZE_PX, SIZE_PX / 2));
	sf::RectangleShape ground_3d(sf::Vector2f(SIZE_PX, SIZE_PX / 2));
	sky.setPosition(SIZE_PX, 0);
	ground_3d.setPosition(SIZE_PX, SIZE_PX / 2);
	sky.setFillColor(sf::Color(50, 150, 230));
	ground_3d.setFillColor(sf::Color(60, 160, 10));
	window.draw(sky);
	window.draw(ground_3d);

	// Draw 3D walls using ray casting results
	for (int i = 0; i < rays.size(); i++) {
		Line ray = rays[i];
		// The shorter/further away the wall, the darker its colour
		int c = map_range(std::sqrt(ray.length), 0, std::sqrt(MAX_RAY_LENGTH), 255, 128);
		int y = (SIZE_PX - ray.proj_height) / 2;  // Centre wall vertically
		sf::RectangleShape wall_segment(sf::Vector2f(WALL_WIDTH, ray.proj_height));
		wall_segment.setPosition(i * WALL_WIDTH + SIZE_PX, y);
		wall_segment.setFillColor(sf::Color(c, c, c));
		window.draw(wall_segment);
	}

	sf::Text left_text("2D bird's-eye view", font, 18);
	sf::FloatRect left_text_rect = left_text.getLocalBounds();
	left_text.setOrigin(int(left_text_rect.left + left_text_rect.width / 2), int(left_text_rect.top + left_text_rect.height / 2));
	left_text.setPosition(int(SIZE_PX * 0.5), 15);
	left_text.setFillColor(sf::Color::Black);
	window.draw(left_text);

	sf::Text right_text("3D POV", font, 18);
	sf::FloatRect right_text_rect = right_text.getLocalBounds();
	right_text.setOrigin(int(right_text_rect.left + right_text_rect.width / 2), int(right_text_rect.top + right_text_rect.height / 2));
	right_text.setPosition(int(SIZE_PX * 1.5), 15);
	right_text.setFillColor(sf::Color::Black);
	window.draw(right_text);

	window.display();
}


int main() {
	window.setFramerateLimit(FPS);
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");

	// Start at top-left, looking towards centre
	player_x = player_y = POINT_RADIUS + 1;  // Border wall width = 1px
	player_heading = 45.0 * M_PI / 180.0;  // Convert 45 deg to rad

	generate_obstacles();

	sf::Keyboard::Key key_pressed = sf::Keyboard::Unknown;
	sf::Event event;
	double dx = 0.0, dy = 0.0;

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::KeyPressed:
					if (event.key.code == sf::Keyboard::R) {  // Reset
						player_x = player_y = POINT_RADIUS + 1;
						player_heading = 45.0 * M_PI / 180.0;
						generate_obstacles();
					} else {
						key_pressed = event.key.code;
					}
					break;
				case sf::Event::KeyReleased:
					key_pressed = sf::Keyboard::Unknown;
					break;
			}
		}

		switch (key_pressed) {
			case sf::Keyboard::W:  // Move forwards
				dx = MOVEMENT_SPEED * cos(player_heading);
				dy = MOVEMENT_SPEED * sin(player_heading);
				break;
			case sf::Keyboard::S:  // Move backwards
				dx = -MOVEMENT_SPEED * cos(player_heading);
				dy = -MOVEMENT_SPEED * sin(player_heading);
				break;
			case sf::Keyboard::A:  // Turn left
				player_heading -= TURNING_SPEED * M_PI / 180.0;
				break;
			case sf::Keyboard::D:  // Turn right
				player_heading += TURNING_SPEED * M_PI / 180.0;
				break;
		}

		if (dx != 0.0 || dy != 0.0) {
			if (POINT_RADIUS + 1 <= player_x + dx && player_x + dx <= SIZE_PX - POINT_RADIUS - 1) {
				if (POINT_RADIUS + 1 <= player_y + dy && player_y + dy <= SIZE_PX - POINT_RADIUS - 1) {
					player_x += dx;
					player_y += dy;
					dx = dy = 0.0;
				}
			}
		}

		generate_rays();
		draw();
	}

	return 0;
}
