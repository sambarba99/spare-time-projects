/*
Drawing with the Discrete Fourier Transform

Controls:
	Right-click: enter/exit drawing mode
	Left-click [and drag]: draw freestyle
	P/G/T: draw preset pi symbol/guitar/T. Rex
	Up/down arrows: increase/decrease number of epicycles to draw with
	Space: toggle animation

Author: Sam Barba
Created 18/11/2022
*/

#include <complex>
#include <iomanip>
#include <SFML/Graphics.hpp>

#include "presets.h"

using std::complex;


const int SIZE = 800;
const int LABEL_HEIGHT = 30;
const int FPS = 60;

struct MyComplex {
	double re;
	double im;
	double frequency;
	double ampltiude;
	double phase;
};
int num_epicycles;
sf::RenderWindow window(sf::VideoMode(SIZE, SIZE + LABEL_HEIGHT), "Drawing with the Discrete Fourier Transform", sf::Style::Close);
sf::Font font;


vector<pair<int, int>> bresenham(int x1, int y1, const int x2, const int y2) {
	// Bresenham's algorithm

	vector<pair<int, int>> result;
	int dx = abs(x1 - x2);
	int dy = -abs(y1 - y2);
	int sx = x1 < x2 ? 1 : -1;
	int sy = y1 < y2 ? 1 : -1;
	int err = dx + dy;
	int e2;

	while (true) {
		result.emplace_back(x1, y1);

		if (x1 == x2 && y1 == y2) return result;

		e2 = 2 * err;
		if (e2 >= dy) {
			err += dy;
			x1 += sx;
		}
		if (e2 <= dx) {
			err += dx;
			y1 += sy;
		}
	}
}


vector<pair<int, int>> interpolate(const vector<pair<int, int>>& coords) {
	if (coords.size() < 2)
		return coords;

	vector<pair<int, int>> interpolated;
	for (int i = 0; i < coords.size() - 1; i++) {
		vector<pair<int, int>> bres_coords = bresenham(coords[i].first, coords[i].second, coords[i + 1].first, coords[i + 1].second);
		for (const auto& c : bres_coords)
			interpolated.emplace_back(c);
	}

	return interpolated;
}


vector<MyComplex> dft(const vector<complex<double>>& x) {
	// Discrete Fourier Transform (see https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Definition)

	int N = x.size();
	vector<MyComplex> X(N);

	for (int k = 0; k < N; k++) {
		complex<double> sum(0.0, 0.0);
		for (int n = 0; n < N; n++) {
			complex<double> temp(0.0, -2.0 * M_PI * k * n / double(N));
			sum += x[n] * exp(temp);
		}
		sum /= double(N);  // Average the sum's contribution over N

		X[k] = {sum.real(), sum.imag(), double(k), abs(sum), atan2(sum.imag(), sum.real())};
	}

	// Descending order of amplitude
	sort(X.begin(), X.end(), [](const MyComplex& lhs, const MyComplex& rhs) {
		return lhs.ampltiude > rhs.ampltiude;
	});
	return X;
}


vector<MyComplex> compute_fourier_from_coords(const vector<pair<int, int>>& drawing_coords) {
	// Centre around origin
	vector<pair<int, int>> centered_coords;
	for (const auto& [x, y] : drawing_coords)
		centered_coords.emplace_back(x - SIZE / 2, y - SIZE / 2);

	// Fill any gaps
	vector<pair<int, int>> interpolated = interpolate(centered_coords);

	// Skip 3 points at a time (don't need that much resolution)
	vector<pair<int, int>> drawing_path;
	for (int i = 0; i < interpolated.size(); i++)
		if (i % 4 == 0)
			drawing_path.emplace_back(interpolated[i]);

	// Convert to complex
	vector<complex<double>> complex_vector;
	for (const auto& [x, y] : drawing_path)
		complex_vector.emplace_back(double(x), double(y));

	vector<MyComplex> fourier = dft(complex_vector);

	return fourier;
}


pair<double, double> epicycles(double x, double y, const vector<MyComplex>& fourier, const double time) {
	double prev_x, prev_y;
	for (int i = 0; i < num_epicycles; i++) {
		prev_x = x;
		prev_y = y;
		double freq = fourier[i].frequency;
		double radius = fourier[i].ampltiude;
		double phase = fourier[i].phase;
		x += radius * cos(freq * time + phase);
		y += radius * sin(freq * time + phase);

		sf::CircleShape circle(radius);
		circle.setPosition(prev_x - radius, prev_y - radius);
		circle.setFillColor(sf::Color(0, 0, 0, 0));
		circle.setOutlineColor(sf::Color(60, 60, 60));
		circle.setOutlineThickness(1.f);

		sf::Vertex line[] = {
			sf::Vertex(sf::Vector2f(prev_x, prev_y)),
			sf::Vertex(sf::Vector2f(x, y))
		};
		window.draw(circle);
		window.draw(line, 2, sf::Lines);
	}

	return {x, y};
}


void draw_label(const std::string label, const int pause_millisecs) {
	sf::RectangleShape lbl_area(sf::Vector2f(SIZE, LABEL_HEIGHT));
	lbl_area.setPosition(0, SIZE);
	lbl_area.setFillColor(sf::Color::Black);
	window.draw(lbl_area);

	sf::Text text(label, font, 14);
	sf::FloatRect text_rect = text.getLocalBounds();
	text.setOrigin(int(text_rect.left + text_rect.width / 2), int(text_rect.top + text_rect.height / 2));
	text.setPosition(SIZE / 2, SIZE + LABEL_HEIGHT / 2);
	text.setFillColor(sf::Color::White);
	window.draw(text);

	if (pause_millisecs) {
		window.display();
		sf::sleep(sf::milliseconds(pause_millisecs));
	}
}


int main() {
	bool paused = false;
	bool left_btn_down = false;
	bool user_drawing_mode = true;
	vector<pair<int, int>> user_drawing_coords;
	vector<MyComplex> fourier;
	vector<pair<double, double>> path;
	double time = 0, dt = 0;
	int screenshot_counter = 0;
	sf::Event event;
	sf::Vector2i mouse_pos;

	window.setFramerateLimit(FPS);
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;

				case sf::Event::MouseButtonPressed:
					if (event.mouseButton.button == sf::Mouse::Left && user_drawing_mode) {
						left_btn_down = true;
					} else if (event.mouseButton.button == sf::Mouse::Right) {
						user_drawing_mode = !user_drawing_mode;
						if (user_drawing_mode) {  // Start drawing
							user_drawing_coords.clear();  // Clear for new drawing
							fourier.clear();  // Clear previous calculations
							path.clear();  // Clear previous renders
							time = 0.0;
						} else {  // Finished drawing
							if (user_drawing_coords.size() < 2) {
								draw_label("Need at least 2 points", 750);
								user_drawing_mode = true;
							} else {
								fourier = compute_fourier_from_coords(user_drawing_coords);
								num_epicycles = fourier.size();
								dt = 2 * M_PI / double(num_epicycles);
								paused = false;
							}
						}
					}
					break;

				case sf::Event::MouseMoved:
					if (left_btn_down && user_drawing_mode) {
						mouse_pos = sf::Mouse::getPosition(window);
						user_drawing_coords.emplace_back(mouse_pos.x, mouse_pos.y);
					}
					break;

				case sf::Event::MouseButtonReleased:
					if (event.mouseButton.button == sf::Mouse::Left) {
						if (left_btn_down && user_drawing_mode) {
							mouse_pos = sf::Mouse::getPosition(window);
							user_drawing_coords.emplace_back(mouse_pos.x, mouse_pos.y);
							left_btn_down = false;
						}
					}
					break;

				case sf::Event::KeyPressed:
					switch (event.key.code) {
						case sf::Keyboard::Up: case sf::Keyboard::Down:
							if (fourier.empty())
								continue;
							if (event.key.code == sf::Keyboard::Up) {
								num_epicycles = std::min(num_epicycles * 2, int(fourier.size()));
							} else {
								int pow2 = pow(2, int(log2(num_epicycles)));
								num_epicycles = pow2 == num_epicycles ? pow2 / 2 : pow2;
								num_epicycles = std::max(num_epicycles, 2);
							}
							if (num_epicycles == fourier.size())
								draw_label("No. epicycles = " + std::to_string(num_epicycles) + " (max)", 500);
							else
								draw_label("No. epicycles = " + std::to_string(num_epicycles), 500);
							path.clear();
							time = 0.0;
							continue;
						case sf::Keyboard::P:
							fourier = compute_fourier_from_coords(PI);
							break;
						case sf::Keyboard::G:
							fourier = compute_fourier_from_coords(GUITAR);
							break;
						case sf::Keyboard::T:
							fourier = compute_fourier_from_coords(T_REX);
							break;
						case sf::Keyboard::Space:
							paused = !paused;
							continue;
					}
					user_drawing_mode = paused = false;
					path.clear();
					time = 0.0;
					num_epicycles = fourier.size();
					dt = 2 * M_PI / double(num_epicycles);
					break;
			}
		}

		if (paused && !user_drawing_mode)
			continue;

		window.clear();

		if (user_drawing_mode) {
			sf::VertexArray pixels(sf::Points);
			for (const auto& [x, y] : user_drawing_coords)
				pixels.append(sf::Vertex(sf::Vector2f(x, y), sf::Color::Red));
			window.draw(pixels);
		} else {  // Draw Fourier result
			pair<double, double> epicycle_final_pos = epicycles(SIZE / 2.0, SIZE / 2.0, fourier, time);
			path.emplace_back(epicycle_final_pos.first, epicycle_final_pos.second);
			for (int i = 0; i < path.size() - 1; i++) {
				sf::Vertex line[] = {
					sf::Vertex(sf::Vector2f(path[i].first, path[i].second), sf::Color::Red),
					sf::Vertex(sf::Vector2f(path[i + 1].first, path[i + 1].second), sf::Color::Red)
				};
				window.draw(line, 2, sf::Lines);
			}

			time += dt;
			if (time > 2 * M_PI) {  // Done a full revolution, so reset
				path.clear();
				time = 0.0;
			}
		}

		if (user_drawing_mode)
			draw_label("Draw something, or select a preset with P/G/T. Right-click to exit drawing mode.", 0);
		window.display();

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
