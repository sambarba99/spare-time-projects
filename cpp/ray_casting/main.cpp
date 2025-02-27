/*
Ray casting demo

Controls:
	WASD: move around
	R: reset

Author: Sam Barba
Created 15/11/2022
*/

#include <cmath>
#include <random>
#include <SFML/Graphics.hpp>

using std::vector;


// World constants
const int WIDTH = 1400;
const int HEIGHT = 700;
const int GRID_ROWS = 10;  // World is a grid
const int GRID_COLS = 20;
const int GRID_SQUARE_SIZE = WIDTH / GRID_COLS;  // HEIGHT / GRID_ROWS
const int BORDER_LIM = 10;

// Ray/rendering constants
const double MAX_RAY_LENGTH = sqrt(pow(WIDTH - 2 * BORDER_LIM, 2) + pow(HEIGHT - 2 * BORDER_LIM, 2));
const int N_RAYS = WIDTH;
const double FOV_ANGLE = M_PI / 3.0;  // 60 deg
const double DELTA_ANGLE = FOV_ANGLE / N_RAYS;
const double SCREEN_DIST = WIDTH * 0.5 / tan(FOV_ANGLE * 0.5);
const int WALL_WIDTH = WIDTH / N_RAYS;  // Width (px) of each wall segment to render in 3D
const int PROJ_HEIGHT_SCALE = 50;
const double MINIMAP_SCALE = 0.2;
const int FPS = 60;

// Player constants
const double MOVEMENT_SPEED = 4.0;
const double TURNING_SPEED = 0.8;

struct Wall {
	double x1;
	double y1;
	double x2;
	double y2;
};

struct Ray {
	double x1;
	double y1;
	double x2;
	double y2;
	double length;
};

vector<Wall> walls;
vector<std::pair<Ray, double>> rayCastingResult;
double playerX, playerY;
double playerHeading;
std::random_device rd;
std::mt19937 gen(rd());
sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Ray casting demo", sf::Style::Close);


vector<Wall> makeBox(const int gridIdx) {
	double xTopLeft = (gridIdx % GRID_COLS) * GRID_SQUARE_SIZE;
	double yTopLeft = (gridIdx / GRID_COLS) * GRID_SQUARE_SIZE;
	double xBottomRight = xTopLeft + GRID_SQUARE_SIZE;
	double yBottomRight = yTopLeft + GRID_SQUARE_SIZE;
	Wall w1 = {xTopLeft, yTopLeft, xBottomRight, yTopLeft};
	Wall w2 = {xBottomRight, yTopLeft, xBottomRight, yBottomRight};
	Wall w3 = {xBottomRight, yBottomRight, xTopLeft, yBottomRight};
	Wall w4 = {xTopLeft, yBottomRight, xTopLeft, yTopLeft};
	return {w1, w2, w3, w4};
}


void generateWalls(const int nBoxes = 15) {
	// Get random grid indices without replacement
	vector<int> allGridIndices;
	for (int i = 0; i < GRID_ROWS * GRID_COLS; i++)
		allGridIndices.push_back(i);
	std::shuffle(allGridIndices.begin(), allGridIndices.end(), gen);

	walls.clear();
	for (int i = 0; i < nBoxes; i++) {
		vector<Wall> boxWalls = makeBox(allGridIndices[i]);
		for (const Wall& w : boxWalls)
			walls.push_back(w);
	}

	// World border
	Wall wb1 = {0, 0, WIDTH, 0};
	Wall wb2 = {WIDTH, 0, WIDTH, HEIGHT};
	Wall wb3 = {0, HEIGHT, WIDTH, HEIGHT};
	Wall wb4 = {1, 0, 1, HEIGHT};
	walls.push_back(wb1);
	walls.push_back(wb2);
	walls.push_back(wb3);
	walls.push_back(wb4);
}


vector<double> findIntersection(const Ray ray, const Wall wall) {
	double rx1 = ray.x1, ry1 = ray.y1;
	double rx2 = ray.x2, ry2 = ray.y2;
	double wx1 = wall.x1, wy1 = wall.y1;
	double wx2 = wall.x2, wy2 = wall.y2;
	
	double denom = (rx1 - rx2) * (wy1 - wy2) - (ry1 - ry2) * (wx1 - wx2);
	if (denom == 0)
		return {-1, -1};
	
	double t = ((rx1 - wx1) * (wy1 - wy2) - (ry1 - wy1) * (wx1 - wx2)) / denom;
	double u = -((rx1 - rx2) * (ry1 - wy1) - (ry1 - ry2) * (rx1 - wx1)) / denom;

	if (0 <= t && t <= 1 && 0 <= u && u <= 1) {
		double intersectionX = rx1 + t * (rx2 - rx1);
		double intersectionY = ry1 + t * (ry2 - ry1);
		return {intersectionX, intersectionY};
	}

	return {-1, -1};
}


void generateRays() {
	rayCastingResult.clear();
	double rayAngle = playerHeading - FOV_ANGLE / 2.0;
	for (int i = 0; i < N_RAYS; i++) {
		double endX = WIDTH * HEIGHT * cos(rayAngle) + playerX;
		double endY = WIDTH * HEIGHT * sin(rayAngle) + playerY;

		// Initialise ray with arbitrary length attribute; this is calculated properly in loop below
		Ray ray = {playerX, playerY, endX, endY, WIDTH * HEIGHT};

		// After this loop, ray.length will be the distance to the nearest wall that the ray hits
		for (const Wall& wall : walls) {
			vector<double> intersection = findIntersection(ray, wall);
			if (intersection[0] != -1) {
				ray.x2 = intersection[0];
				ray.y2 = intersection[1];
				ray.length = sqrt(pow(ray.x1 - ray.x2, 2) + pow(ray.y1 - ray.y2, 2));
			}
		}

		double disortedDist = ray.length;
		double correctDist = disortedDist * cos(playerHeading - rayAngle);  // Remove fish eye distortion
		correctDist += 1e-6;  // Avoid division by 0
		double projHeight = SCREEN_DIST / correctDist * PROJ_HEIGHT_SCALE;
		projHeight = std::min(double(HEIGHT), projHeight);

		rayCastingResult.push_back({ray, projHeight});
		rayAngle += DELTA_ANGLE;
	}
}


double mapRange(const double x, const double fromLo, const double fromHi, const double toLo, const double toHi) {
	// Map x from [fromLo, fromHi] to [toLo, toHi]
	if (fromHi - fromLo == 0) return toHi;
	return (x - fromLo) / (fromHi - fromLo) * (toHi - toLo) + toLo;
}


void drawPovMode() {
	// Draw sky and ground
	sf::RectangleShape sky(sf::Vector2f(WIDTH, HEIGHT / 2));
	sf::RectangleShape ground(sf::Vector2f(WIDTH, HEIGHT / 2));
	sky.setPosition(0, 0);
	ground.setPosition(0, HEIGHT / 2);
	sky.setFillColor(sf::Color(20, 100, 255));
	ground.setFillColor(sf::Color(20, 150, 20));
	window.draw(sky);
	window.draw(ground);

	// Draw walls
	for (int i = 0; i < rayCastingResult.size(); i++) {
		Ray ray = rayCastingResult[i].first;
		double projHeight = rayCastingResult[i].second;
		int c = mapRange(sqrt(ray.length), 0, sqrt(MAX_RAY_LENGTH), 255, 50);
		int y = (HEIGHT - projHeight) / 2;  // Centre wall vertically
		sf::RectangleShape wallStrip(sf::Vector2f(WALL_WIDTH, projHeight));
		wallStrip.setPosition(i * WALL_WIDTH, y);
		wallStrip.setFillColor(sf::Color(c, c, c));
		window.draw(wallStrip);
	}
}


void drawMiniMap() {
	// 2D rays
	for (int i = 0; i < rayCastingResult.size(); i++) {
		Ray r = rayCastingResult[i].first;
		if (i % 40 == 0) {
			sf::Vertex line[] = {
				sf::Vertex(sf::Vector2f(r.x1 * MINIMAP_SCALE, r.y1 * MINIMAP_SCALE), sf::Color::Red),
				sf::Vertex(sf::Vector2f(r.x2 * MINIMAP_SCALE, r.y2 * MINIMAP_SCALE), sf::Color::Red)
			};
			window.draw(line, 2, sf::Lines);
		}
	}

	// Player
	sf::CircleShape circle(4.f);
	circle.setPosition(playerX * MINIMAP_SCALE - 4, playerY * MINIMAP_SCALE - 4);
	circle.setFillColor(sf::Color::Black);
	window.draw(circle);

	// 2D walls
	for (const Wall& w : walls) {
		sf::Vertex line[] = {
			sf::Vertex(sf::Vector2f(w.x1 * MINIMAP_SCALE, w.y1 * MINIMAP_SCALE), sf::Color::Black),
			sf::Vertex(sf::Vector2f(w.x2 * MINIMAP_SCALE, w.y2 * MINIMAP_SCALE), sf::Color::Black)
		};
		window.draw(line, 2, sf::Lines);
	}
}


int main() {
	window.setFramerateLimit(FPS);

	// Start at top-left, looking towards centre
	playerX = playerY = BORDER_LIM;
	playerHeading = atan2(HEIGHT, WIDTH);

	generateWalls();
	generateRays();
	drawPovMode();
	drawMiniMap();

	sf::Keyboard::Key keyPressed = sf::Keyboard::Unknown;
	sf::Event event;
	double dx, dy;

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::KeyPressed:
					if (event.key.code == sf::Keyboard::R) {  // Reset
						playerX = playerY = BORDER_LIM;
						playerHeading = atan2(HEIGHT, WIDTH);
						generateWalls();
					} else {
						keyPressed = event.key.code;
					}
					break;
				case sf::Event::KeyReleased:
					keyPressed = sf::Keyboard::Unknown;
					break;
			}
		}

		switch (keyPressed) {
			case sf::Keyboard::W:  // Move forwards
				dx = MOVEMENT_SPEED * cos(playerHeading);
				dy = MOVEMENT_SPEED * sin(playerHeading);
				if (BORDER_LIM <= playerX + dx && playerX + dx < WIDTH - BORDER_LIM) {
					if (BORDER_LIM <= playerY + dy && playerY + dy < HEIGHT - BORDER_LIM) {
						playerX += dx;
						playerY += dy;
					}
				}
				break;
			case sf::Keyboard::S:  // Move backwards
				dx = MOVEMENT_SPEED * cos(playerHeading);
				dy = MOVEMENT_SPEED * sin(playerHeading);
				if (BORDER_LIM <= playerX - dx && playerX - dx < WIDTH - BORDER_LIM) {
					if (BORDER_LIM <= playerY - dy && playerY - dy < HEIGHT - BORDER_LIM) {
						playerX -= dx;
						playerY -= dy;
					}
				}
				break;
			case sf::Keyboard::A:  // Turn left
				playerHeading -= TURNING_SPEED * M_PI / 180.0;
				break;
			case sf::Keyboard::D:  // Turn right
				playerHeading += TURNING_SPEED * M_PI / 180.0;
				break;
		}

		generateRays();
		drawPovMode();
		drawMiniMap();
		window.display();
	}

	return 0;
}
