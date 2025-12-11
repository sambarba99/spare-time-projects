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


pair<float, float> linear_interpolate(const pair<float, float>& a, const pair<float, float>& b, const float t) {
	// Linear interpolation between (ax,ay) and (bx,by) by amount t

	float ax = a.first, ay = a.second;
	float bx = b.first, by = b.second;
	float lx = ax + t * (bx - ax);
	float ly = ay + t * (by - ay);
	return {lx, ly};
}


pair<float, float> bezier_point(vector<pair<float, float>> control_points, const float t) {
	while (control_points.size() > 1) {
		vector<pair<float, float>> control_points_temp(control_points.size() - 1);
		for (int i = 0; i < control_points.size() - 1; i++) {
			pair<float, float> lerp = linear_interpolate(control_points[i], control_points[i + 1], t);
			control_points_temp[i] = lerp;
		}
		control_points = control_points_temp;
	}

	return control_points[0];
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
			pair<float, float> point = bezier_point(points, t);
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


vector<float> calc_point_distances_from_mouse(const int mouse_x, const int mouse_y) {
	vector<float> distances;
	distances.reserve(points.size());
	for (const auto& [x, y] : points)
		distances.push_back(std::hypot(x - mouse_x, y - mouse_y));
	return distances;
}


int main() {
	window.setFramerateLimit(FPS);
	window.display();

	int clicked_point_idx = -1;
	bool left_btn_down = false;
	sf::Vector2i mouse_pos;
	sf::Event event;

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;

				case sf::Event::MouseButtonPressed:
					mouse_pos = sf::Mouse::getPosition(window);

					if (event.mouseButton.button == sf::Mouse::Left) {  // Left-click to drag point
						if (points.empty())
							continue;

						left_btn_down = true;
						vector<float> point_distances_from_mouse = calc_point_distances_from_mouse(mouse_pos.x, mouse_pos.y);
						float min_dist = *min_element(point_distances_from_mouse.begin(), point_distances_from_mouse.end());
						if (min_dist <= POINT_RADIUS)  // Mouse is on a point
							clicked_point_idx = find(point_distances_from_mouse.begin(), point_distances_from_mouse.end(), min_dist) - point_distances_from_mouse.begin();
						else
							clicked_point_idx = -1;
					} else if (event.mouseButton.button == sf::Mouse::Right) {  // Right-click to add a point
						if (points.size() < MAX_POINTS) {
							points.emplace_back(mouse_pos.x, mouse_pos.y);
							draw();
						}
					}
					break;

				case sf::Event::MouseMoved:
					mouse_pos = sf::Mouse::getPosition(window);
					break;

				case sf::Event::MouseButtonReleased:
					left_btn_down = false;
					break;

				case sf::Event::KeyPressed:
					if (event.key.code == sf::Keyboard::R) {  // Reset
						points.clear();
						draw();
					}
					break;
			}
		}

		if (left_btn_down && clicked_point_idx != -1) {
			points[clicked_point_idx] = {mouse_pos.x, mouse_pos.y};
			draw();
		}
	}

	return 0;
}
