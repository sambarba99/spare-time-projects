/*
Ray casting demo

Controls:
	WASD: move around
	R: reset

Author: Sam Barba
Created 15/11/2022
*/

#include <cmath>
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
const double MAX_RAY_LENGTH = sqrt(2 * SIZE_PX * SIZE_PX);
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
	double projHeight;
};

struct Circle {
	double x;
	double y;
	double r;
};


vector<Line> rays;
vector<Line> walls;
vector<Circle> circles;
double playerX, playerY, playerHeading;
std::random_device rd;
std::mt19937 gen(rd());
sf::RenderWindow window(sf::VideoMode(SIZE_PX * 2, SIZE_PX), "Ray casting demo", sf::Style::Close);
sf::Font font;


vector<Line> makeBox(const int gridIdx) {
	double x1 = (gridIdx % SIZE_GRID) * GRID_SQUARE_SIZE;
	double y1 = (gridIdx / SIZE_GRID) * GRID_SQUARE_SIZE;
	double x2 = x1 + GRID_SQUARE_SIZE;
	double y2 = y1 + GRID_SQUARE_SIZE;
	Line northWall = {x1, y1, x2, y1};
	Line eastWall = {x2, y1, x2, y2};
	Line southWall = {x2, y2, x1, y2};
	Line westWall = {x1, y2, x1, y1};
	return {northWall, eastWall, southWall, westWall};
}


void generateObstacles(const int numBoxes = 5, const int numCircles = 5) {
	// Shuffle all grid indices for sampling
	vector<int> allGridIndices(SIZE_GRID * SIZE_GRID);
	std::iota(allGridIndices.begin(), allGridIndices.end(), 0);
	std::shuffle(allGridIndices.begin(), allGridIndices.end(), gen);

	walls.clear();
	for (int i = 0; i < numBoxes; i++) {
		vector<Line> boxWalls = makeBox(allGridIndices[i]);
		for (const Line& wall : boxWalls)
			walls.push_back(wall);
	}

	circles.clear();
	for (int i = numBoxes; i < numBoxes + numCircles; i++) {
		int gridIdx = allGridIndices[i];
		double cx = (gridIdx % SIZE_GRID) * GRID_SQUARE_SIZE + GRID_SQUARE_SIZE / 2;
		double cy = (gridIdx / SIZE_GRID) * GRID_SQUARE_SIZE + GRID_SQUARE_SIZE / 2;
		circles.push_back({cx, cy, GRID_SQUARE_SIZE / 2});
	}

	// World border
	Line northBorder = {0, 0, SIZE_PX, 0};
	Line eastBorder = {SIZE_PX, 0, SIZE_PX, SIZE_PX};
	Line southBorder = {0, SIZE_PX - 1, SIZE_PX, SIZE_PX - 1};
	Line westBorder = {1, 0, 1, SIZE_PX};
	walls.insert(walls.end(), {northBorder, eastBorder, southBorder, westBorder});
}


std::optional<vector<double>> findRayWallIntersection(const Line& ray, const Line& wall) {
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
		double intersectionX = rx1 + t * (rx2 - rx1);
		double intersectionY = ry1 + t * (ry2 - ry1);
		return vector<double>{intersectionX, intersectionY};
	}

	return std::nullopt;
}


std::optional<vector<double>> findRayCircleIntersection(const Line& ray, const Circle& circle) {
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

	double sqrtD = sqrt(discriminant);
	double t1 = (-b - sqrtD) / (2 * a);
	double t2 = (-b + sqrtD) / (2 * a);

	vector<double> validTs;
	if (0 <= t1 && t1 <= 1)
		validTs.push_back(t1);
	if (0 <= t2 && t2 <= 1)
		validTs.push_back(t2);
	if (validTs.empty())
		return std::nullopt;

	double t = *std::min_element(validTs.begin(), validTs.end());
	double intersectionX = rx1 + t * dx;
	double intersectionY = ry1 + t * dy;

	return vector<double>{intersectionX, intersectionY};
}


void generateRays() {
	rays.clear();
	double rayAngle = playerHeading - FOV_ANGLE / 2.0;
	double endX, endY, ix, iy, length, correctedDist, projHeight;

	for (int i = 0; i < NUM_RAYS; i++) {
		// Initialise ray arbitrarily long (2 * SIZE_PX ensures it goes off the screen).
		// After the following loops, ray.length will be the distance to the nearest object that the ray hits.
		endX = 2 * SIZE_PX * cos(rayAngle) + playerX;
		endY = 2 * SIZE_PX * sin(rayAngle) + playerY;
		Line ray = {playerX, playerY, endX, endY};

		for (const Line& wall : walls) {
			std::optional<vector<double>> intersection = findRayWallIntersection(ray, wall);
			if (intersection) {
				ray.x2 = (*intersection)[0];
				ray.y2 = (*intersection)[1];
			}
		}

		for (const Circle& circle : circles) {
			std::optional<vector<double>> intersection = findRayCircleIntersection(ray, circle);
			if (intersection) {
				ray.x2 = (*intersection)[0];
				ray.y2 = (*intersection)[1];
			}
		}

		ray.length = sqrt(pow(ray.x1 - ray.x2, 2) + pow(ray.y1 - ray.y2, 2));
		correctedDist = ray.length * cos(playerHeading - rayAngle);  // Remove fisheye distortion
		correctedDist += 1e-6;  // Avoid division by 0
		projHeight = SCREEN_DIST / correctedDist * PROJ_HEIGHT_SCALE;
		ray.projHeight = std::min(projHeight, double(SIZE_PX));

		rays.push_back(ray);
		rayAngle += DELTA_ANGLE;
	}
}


double mapRange(const double x, const double fromLo, const double fromHi, const double toLo, const double toHi) {
	// Map x from [fromLo, fromHi] to [toLo, toHi]
	if (fromHi - fromLo == 0)
		return toHi;
	return (x - fromLo) / (fromHi - fromLo) * (toHi - toLo) + toLo;
}


void draw() {
	window.clear();

	// ---------- Left half of screen (bird's-eye view) ----------

	// Ground
	sf::RectangleShape ground2d(sf::Vector2f(SIZE_PX, SIZE_PX));
	ground2d.setPosition(0, 0);
	ground2d.setFillColor(sf::Color(160, 160, 160));
	window.draw(ground2d);

	// 2D rays
	for (int i = 0; i < rays.size(); i++)
		if (i % 20 == 0) {
			Line ray = rays[i];
			sf::Vertex line[] = {
				sf::Vertex(sf::Vector2f(ray.x1, ray.y1), sf::Color::Red),
				sf::Vertex(sf::Vector2f(ray.x2, ray.y2), sf::Color::Red)
			};
			window.draw(line, 2, sf::Lines);
		}

	// Player
	sf::CircleShape circle(POINT_RADIUS);
	circle.setPosition(playerX - POINT_RADIUS, playerY - POINT_RADIUS);
	circle.setFillColor(sf::Color::Black);
	window.draw(circle);

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
	sf::RectangleShape ground3d(sf::Vector2f(SIZE_PX, SIZE_PX / 2));
	sky.setPosition(SIZE_PX, 0);
	ground3d.setPosition(SIZE_PX, SIZE_PX / 2);
	sky.setFillColor(sf::Color(50, 150, 230));
	ground3d.setFillColor(sf::Color(60, 160, 10));
	window.draw(sky);
	window.draw(ground3d);

	// Draw 3D walls using ray casting results
	for (int i = 0; i < rays.size(); i++) {
		Line ray = rays[i];
		// The shorter/further away the wall, the darker its colour
		int c = mapRange(sqrt(ray.length), 0, sqrt(MAX_RAY_LENGTH), 255, 128);
		int y = (SIZE_PX - ray.projHeight) / 2;  // Centre wall vertically
		sf::RectangleShape wallSegment(sf::Vector2f(WALL_WIDTH, ray.projHeight));
		wallSegment.setPosition(i * WALL_WIDTH + SIZE_PX, y);
		wallSegment.setFillColor(sf::Color(c, c, c));
		window.draw(wallSegment);
	}

	sf::Text leftText("2D bird's-eye view", font, 18);
	sf::FloatRect leftTextRect = leftText.getLocalBounds();
	leftText.setOrigin(leftTextRect.left + leftTextRect.width / 2, leftTextRect.top + leftTextRect.height / 2);
	leftText.setPosition(SIZE_PX * 0.5, 15);
	leftText.setFillColor(sf::Color::Black);
	window.draw(leftText);

	sf::Text rightText("3D POV", font, 18);
	sf::FloatRect rightTextRect = rightText.getLocalBounds();
	rightText.setOrigin(rightTextRect.left + rightTextRect.width / 2, rightTextRect.top + rightTextRect.height / 2);
	rightText.setPosition(SIZE_PX * 1.5, 15);
	rightText.setFillColor(sf::Color::Black);
	window.draw(rightText);

	window.display();
}


int main() {
	window.setFramerateLimit(FPS);
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");

	// Start at top-left, looking towards centre
	playerX = playerY = POINT_RADIUS + 1;  // Border wall width = 1px
	playerHeading = 45.0 * M_PI / 180.0;  // Convert 45 deg to rad

	generateObstacles();

	sf::Keyboard::Key keyPressed = sf::Keyboard::Unknown;
	sf::Event event;
	double dx = 0.0, dy = 0.0;

	while (window.isOpen()) {
		while (window.pollEvent(event))
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::KeyPressed:
					if (event.key.code == sf::Keyboard::R) {  // Reset
						playerX = playerY = POINT_RADIUS + 1;
						playerHeading = 45.0 * M_PI / 180.0;
						generateObstacles();
					} else {
						keyPressed = event.key.code;
					}
					break;
				case sf::Event::KeyReleased:
					keyPressed = sf::Keyboard::Unknown;
					break;
			}

		switch (keyPressed) {
			case sf::Keyboard::W:  // Move forwards
				dx = MOVEMENT_SPEED * cos(playerHeading);
				dy = MOVEMENT_SPEED * sin(playerHeading);
				break;
			case sf::Keyboard::S:  // Move backwards
				dx = -MOVEMENT_SPEED * cos(playerHeading);
				dy = -MOVEMENT_SPEED * sin(playerHeading);
				break;
			case sf::Keyboard::A:  // Turn left
				playerHeading -= TURNING_SPEED * M_PI / 180.0;
				break;
			case sf::Keyboard::D:  // Turn right
				playerHeading += TURNING_SPEED * M_PI / 180.0;
				break;
		}
		if (dx != 0.0 || dy != 0.0)
			if (POINT_RADIUS + 1 <= playerX + dx && playerX + dx <= SIZE_PX - POINT_RADIUS - 1)
				if (POINT_RADIUS + 1 <= playerY + dy && playerY + dy <= SIZE_PX - POINT_RADIUS - 1) {
					playerX += dx;
					playerY += dy;
					dx = dy = 0.0;
				}

		generateRays();
		draw();
	}

	return 0;
}
