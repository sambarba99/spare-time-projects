/*
3D projection demo

Controls:
	Q/A: x translation
	W/S: y translation
	E/D: z translation
	Z/X: x rotation
	C/V: y rotation
	B/N: z rotation
	T: toggle axis rendering
	R: reset

Author: Sam Barba
Created 03/02/2026
*/

#include <array>
#include <cmath>
#include <SFML/Graphics.hpp>

using std::vector;


using Mat = std::array<std::array<double, 3>, 3>;
struct Point2D {
	double x, y;
};
struct Point3D {
	double x, y, z;
};
const int SIZE = 600;
const int FPS = 60;
const double MOVEMENT_SPEED = 0.01;
const double ROTATION_SPEED = 0.02;
const vector<Point3D> VERTICES = {
	// Face points
	{0.25, 0.25, 0.25},
    {-0.25, 0.25, 0.25},
    {-0.25, -0.25, 0.25},
    {0.25, -0.25, 0.25},
    {0.25, 0.25, -0.25},
    {-0.25, 0.25, -0.25},
    {-0.25, -0.25, -0.25},
    {0.25, -0.25, -0.25},
	// Axis points
	{0.5, 0.0, 0.0},
	{-0.5, 0.0, 0.0},
	{0.0, 0.5, 0.0},
	{0.0, -0.5, 0.0},
	{0.0, 0.0, 0.5},
	{0.0, 0.0, -0.5}
};
const vector<vector<int>> EDGE_LIST = {
	// Connect these vertices to form faces
	{0, 1, 2, 3},
	{4, 5, 6, 7},
	{0, 4},
	{1, 5},
	{2, 6},
	{3, 7},
	// Connect these vertices to form axes
	{8, 9},
	{10, 11},
	{12, 13}
};

double dx = 0;
double dy = 0;
double dz = 1;
double theta_x = 0;
double theta_y = 0;
double theta_z = 0;
Mat orientation = {{  // Identity matrix initially
	{1, 0, 0},
	{0, 1, 0},
	{0, 0, 1}
}};
bool draw_axes = true;
sf::RenderWindow window(sf::VideoMode(SIZE, SIZE), "3D projection demo", sf::Style::Close);
sf::Font font;


Mat multiply(const Mat& a, const Mat& b) {
    Mat r;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    return r;
}


Point3D apply_orientation_matrix(const Point3D& p) {
    return {
        orientation[0][0] * p.x + orientation[0][1] * p.y + orientation[0][2] * p.z,
        orientation[1][0] * p.x + orientation[1][1] * p.y + orientation[1][2] * p.z,
        orientation[2][0] * p.x + orientation[2][1] * p.y + orientation[2][2] * p.z
    };
}


Mat rot_matrix_x(double theta) {
    double c = cos(theta);
	double s = sin(theta);
    return {{
		{1, 0, 0},
		{0, c, -s},
		{0, s, c}
	}};
}


Mat rot_matrix_y(double theta) {
    double c = cos(theta);
	double s = sin(theta);
    return {{
		{c, 0, s},
		{0, 1, 0},
		{-s, 0, c}
	}};
}


Mat rot_matrix_z(double theta) {
    double c = cos(theta);
	double s = sin(theta);
    return {{
		{c, -s, 0},
		{s, c, 0},
		{0, 0, 1}
	}};
}


Point3D translate(const Point3D& p) {
	return {p.x + dx, p.y + dy, p.z + dz};
}


Point2D project(const Point3D& point3d) {
	return {
		point3d.x / point3d.z,
		point3d.y / point3d.z
	};
}


Point2D screen(const Point2D& point2d) {
	// Given a normalised (roughly in [-1, 1]) 2D point from project(), convert it to actual pixel coordinates
	return {
		(point2d.x + 1.0) / 2.0 * SIZE,
		(1.0 - (point2d.y + 1.0) / 2.0) * SIZE
	};
}


void draw() {
	window.clear();

	for (int i = 0; i < EDGE_LIST.size(); i++) {
		if (i > 5 && !draw_axes)
			break;

		for (int j = 0; j < EDGE_LIST[i].size(); j++) {
			vector<int> edges = EDGE_LIST[i];
			int vertex_start_idx = edges[j];
			int vertex_end_idx = edges[(j + 1) % edges.size()];

			Point3D a = apply_orientation_matrix(VERTICES[vertex_start_idx]);
			Point3D b = apply_orientation_matrix(VERTICES[vertex_end_idx]);

			Point2D a_screen = screen(project(translate(a)));
			Point2D b_screen = screen(project(translate(b)));

			sf::Color line_colour = sf::Color::White;
			if (i == 6)
				line_colour = sf::Color::Red;
			else if (i == 7)
				line_colour = sf::Color::Green;
			else if (i == 8)
				line_colour = sf::Color(0, 80, 255);

			sf::Vertex line[] = {
				sf::Vertex(sf::Vector2f(a_screen.x, a_screen.y), line_colour),
				sf::Vertex(sf::Vector2f(b_screen.x, b_screen.y), line_colour)
			};
			window.draw(line, 2, sf::Lines);
		}
	}

	if (draw_axes) {
		sf::Text text("x", font, 18);
		text.setPosition(276, 10);
		text.setFillColor(sf::Color::Red);
		window.draw(text);
		text.setString("y");
		text.setPosition(296, 10);
		text.setFillColor(sf::Color::Green);
		window.draw(text);
		text.setString("z");
		text.setPosition(316, 10);
		text.setFillColor(sf::Color(0, 80, 255));
		window.draw(text);
	}

	window.display();
}


int main() {
	window.setFramerateLimit(FPS);
	sf::Event event;
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;

				case sf::Event::KeyPressed:
					if (event.key.code == sf::Keyboard::T) {
						draw_axes = !draw_axes;
					} else if (event.key.code == sf::Keyboard::R) {
						dx = dy = 0;
						dz = 1;
						theta_x = theta_y = theta_z = 0;
						orientation = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
						draw_axes = true;
					}
					break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q))
			dx += MOVEMENT_SPEED;
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			dx -= MOVEMENT_SPEED;
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
			dy += MOVEMENT_SPEED;
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
			dy -= MOVEMENT_SPEED;
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::E))
			dz += MOVEMENT_SPEED;
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::D) && dz > 1)
			dz -= MOVEMENT_SPEED;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Z))
			orientation = multiply(orientation, rot_matrix_x(ROTATION_SPEED));
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::X))
			orientation = multiply(orientation, rot_matrix_x(-ROTATION_SPEED));
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::C))
			orientation = multiply(orientation, rot_matrix_y(ROTATION_SPEED));
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::V))
			orientation = multiply(orientation, rot_matrix_y(-ROTATION_SPEED));
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::B))
			orientation = multiply(orientation, rot_matrix_z(ROTATION_SPEED));
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::N))
			orientation = multiply(orientation, rot_matrix_z(-ROTATION_SPEED));

		draw();
	}

	return 0;
}
