/*
Bézier curve demo

Controls:
	Right-click: add a point
	Left-click and drag: move a point
	R: reset

Author: Sam Barba
Created 19/11/2022
*/

#include <cmath>
#include <SFML/Graphics.hpp>

using std::pair;
using std::vector;


const int SIZE = 600;
const int MAX_POINTS = 10;
const float POINT_RADIUS = 6.f;
const int FPS = 60;

vector<pair<float, float>> points;
sf::RenderWindow window(sf::VideoMode(SIZE, SIZE), L"Bézier curve drawing", sf::Style::Close);


pair<float, float> linearInterpolate(const pair<float, float>& a, const pair<float, float>& b, const float t) {
	// Linear interpolation between (ax,ay) and (bx,by) by amount t

	float ax = a.first, ay = a.second;
	float bx = b.first, by = b.second;
	float lx = ax + t * (bx - ax);
	float ly = ay + t * (by - ay);
	return {lx, ly};
}


pair<float, float> bezierPoint(vector<pair<float, float>> controlPoints, const float t) {
	while (controlPoints.size() > 1) {
		vector<pair<float, float>> controlPointsTemp(controlPoints.size() - 1);
		for (int i = 0; i < controlPoints.size() - 1; i++) {
			pair<float, float> lerp = linearInterpolate(controlPoints[i], controlPoints[i + 1], t);
			controlPointsTemp[i] = lerp;
		}
		controlPoints = controlPointsTemp;
	}

	return controlPoints[0];
}


void draw() {
	window.clear();

	// Draw connective lines, then curve on top, then points on top

	if (points.size() > 1) {
		for (int i = 0; i < points.size() - 1; i++) {
			sf::Vertex line[] = {
				sf::Vertex(sf::Vector2f(points[i].first, points[i].second), sf::Color(80, 80, 80)),
				sf::Vertex(sf::Vector2f(points[i + 1].first, points[i + 1].second), sf::Color(80, 80, 80))
			};
			window.draw(line, 2, sf::Lines);
		}

		sf::VertexArray pixels(sf::Points);
		for (float t = 0; t <= 1; t += 1e-4) {
			pair<float, float> point = bezierPoint(points, t);
			pixels.append(sf::Vertex(sf::Vector2f(point.first, point.second), sf::Color::White));
		}
		window.draw(pixels);
	}

	for (const auto& [x, y] : points) {
		sf::CircleShape circle(POINT_RADIUS);
		circle.setPosition(x - POINT_RADIUS, y - POINT_RADIUS);
		circle.setFillColor(sf::Color(230, 20, 20));
		window.draw(circle);
	}

	window.display();
}


vector<float> calcPointDistancesFromMouse(const int mouseX, const int mouseY) {
	vector<float> distances;
	for (const auto& [x, y] : points)
		distances.push_back(sqrt(pow(x - mouseX, 2) + pow(y - mouseY, 2)));
	return distances;
}


int main() {
	window.setFramerateLimit(FPS);
	window.display();

	int clickedPointIdx = -1;
	sf::Vector2i mousePos;
	float mouseX, mouseY;
	bool leftBtnDown = false;
	sf::Event event;

	while (window.isOpen()) {
		while (window.pollEvent(event))
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::MouseButtonPressed:
					mousePos = sf::Mouse::getPosition(window);
					mouseX = mousePos.x;
					mouseY = mousePos.y;

					if (event.mouseButton.button == sf::Mouse::Left) {  // Left-click to drag point
						if (points.empty()) continue;

						leftBtnDown = true;
						vector<float> pointDistancesFromMouse = calcPointDistancesFromMouse(mouseX, mouseY);
						float minDist = *min_element(pointDistancesFromMouse.begin(), pointDistancesFromMouse.end());
						if (minDist <= POINT_RADIUS)  // Mouse is on a point
							clickedPointIdx = find(pointDistancesFromMouse.begin(), pointDistancesFromMouse.end(), minDist) - pointDistancesFromMouse.begin();
						else
							clickedPointIdx = -1;
					} else if (event.mouseButton.button == sf::Mouse::Right) {  // Right-click to add a point
						if (points.size() < MAX_POINTS) {
							points.push_back({mouseX, mouseY});
							draw();
						}
					}
					break;
				case sf::Event::MouseMoved:
					mousePos = sf::Mouse::getPosition(window);
					mouseX = mousePos.x;
					mouseY = mousePos.y;
					break;
				case sf::Event::MouseButtonReleased:
					leftBtnDown = false;
					break;
				case sf::Event::KeyPressed:
					if (event.key.code == sf::Keyboard::R) {  // Reset
						points.clear();
						draw();
					}
					break;
			}

		if (leftBtnDown && clickedPointIdx != -1) {
			points[clickedPointIdx] = {mouseX, mouseY};
			draw();
		}
	}

	return 0;
}
