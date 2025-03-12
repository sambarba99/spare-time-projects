/*
Drawing with the Discrete Fourier Transform

Controls:
	Right-click: enter/exit drawing mode
	Left-click [and drag]: draw freestyle
	P/G/T: draw preset pi symbol/guitar/T. Rex
	Up/down arrow: increase/decrease number of epicycles to draw with
	Space: toggle animation

Author: Sam Barba
Created 18/11/2022
*/

#include <complex>
#include <iomanip>
#include <SFML/Graphics.hpp>

#include "presets.h"

using std::complex;
using std::vector;


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
int numEpicycles;
sf::RenderWindow window(sf::VideoMode(SIZE, SIZE + LABEL_HEIGHT), "Drawing with the Discrete Fourier Transform", sf::Style::Close);
sf::Font font;
int screenshotCounter = 0;


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
		result.push_back({x1, y1});

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
	if (coords.size() < 2) return coords;

	vector<pair<int, int>> interpolated;
	for (int i = 0; i < coords.size() - 1; i++) {
		vector<pair<int, int>> bresCoords = bresenham(coords[i].first, coords[i].second, coords[i + 1].first, coords[i + 1].second);
		for (const pair<int, int>& c : bresCoords)
			interpolated.push_back(c);
	}

	return interpolated;
}


vector<MyComplex> dft(const vector<complex<double>>& x) {
	// Discrete Fourier Transform (see https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Definition)

	int N = x.size();
	vector<MyComplex> X;

	for (int k = 0; k < N; k++) {
		complex<double> sum(0.0, 0.0);
		for (int n = 0; n < N; n++) {
			complex<double> temp(0.0, -2.0 * M_PI * k * n / double(N));
			sum += x[n] * exp(temp);
		}
		sum /= double(N);  // Average the sum's contribution over N

		MyComplex Xk = {sum.real(), sum.imag(), double(k), abs(sum), atan2(sum.imag(), sum.real())};
		X.push_back(Xk);
	}

	// Descending order of amplitude
	sort(X.begin(), X.end(), [](const MyComplex& lhs, const MyComplex& rhs) {
		return lhs.ampltiude > rhs.ampltiude;
	});
	return X;
}


vector<MyComplex> computeFourierFromCoords(const vector<pair<int, int>>& drawingCoords) {
	// Centre around origin
	vector<pair<int, int>> centeredCoords;
	for (const pair<int, int>& coords : drawingCoords)
		centeredCoords.push_back({coords.first - SIZE / 2, coords.second - SIZE / 2});

	// Fill any gaps
	vector<pair<int, int>> interpolated = interpolate(centeredCoords);

	// Skip 3 points at a time (don't need that much resolution)
	vector<pair<int, int>> drawingPath;
	for (int i = 0; i < interpolated.size(); i++)
		if (i % 4 == 0)
			drawingPath.push_back(interpolated[i]);

	// Convert to complex
	vector<complex<double>> complexVector;
	for (const pair<int, int>& coords : drawingPath)
		complexVector.push_back({double(coords.first), double(coords.second)});

	vector<MyComplex> fourier = dft(complexVector);

	return fourier;
}


pair<double, double> epicycles(double x, double y, const vector<MyComplex>& fourier, const double time) {
	double prevX, prevY;
	for (int i = 0; i < numEpicycles; i++) {
		prevX = x;
		prevY = y;
		double freq = fourier[i].frequency;
		double radius = fourier[i].ampltiude;
		double phase = fourier[i].phase;
		x += radius * cos(freq * time + phase);
		y += radius * sin(freq * time + phase);

		sf::CircleShape circle(radius);
		circle.setPosition(prevX - radius, prevY - radius);
		circle.setFillColor(sf::Color(0, 0, 0, 0));
		circle.setOutlineColor(sf::Color(60, 60, 60));
		circle.setOutlineThickness(1.f);

		sf::Vertex line[] = {
			sf::Vertex(sf::Vector2f(prevX, prevY)),
			sf::Vertex(sf::Vector2f(x, y))
		};
		window.draw(circle);
		window.draw(line, 2, sf::Lines);
	}

	return {x, y};
}


void drawLabel(const std::string label, const int pauseMilliseconds) {
	sf::RectangleShape lblArea(sf::Vector2f(SIZE, LABEL_HEIGHT));
	lblArea.setPosition(0, SIZE);
	lblArea.setFillColor(sf::Color::Black);
	window.draw(lblArea);

	sf::Text text(label, font, 14);
	sf::FloatRect textRect = text.getLocalBounds();
	text.setOrigin(int(textRect.left + textRect.width / 2), int(textRect.top + textRect.height / 2));
	text.setPosition(SIZE / 2, SIZE + LABEL_HEIGHT / 2);
	text.setFillColor(sf::Color::White);
	window.draw(text);

	if (pauseMilliseconds) {
		window.display();
		sf::sleep(sf::milliseconds(pauseMilliseconds));
	}
}


int main() {
	window.setFramerateLimit(FPS);
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");

	bool leftBtnDown = false;
	bool userDrawingMode = true;
	vector<pair<int, int>> userDrawingCoords;
	vector<MyComplex> fourier;
	vector<pair<double, double>> path;
	double time = 0;
	double dt = 0;
	bool paused = false;
	sf::Event event;

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::MouseButtonPressed:
					if (event.mouseButton.button == sf::Mouse::Left) {
						if (userDrawingMode) leftBtnDown = true;
					} else if (event.mouseButton.button == sf::Mouse::Right) {
						userDrawingMode = !userDrawingMode;
						if (userDrawingMode) {  // Start drawing
							userDrawingCoords.clear();  // Clear for new drawing
							fourier.clear();  // Clear previous calculations
							path.clear();  // Clear previous renders
							time = 0.0;
						} else {  // Finished drawing
							if (userDrawingCoords.size() < 2) {
								drawLabel("Need at least 2 points", 750);
								userDrawingMode = true;
							} else {
								fourier = computeFourierFromCoords(userDrawingCoords);
								numEpicycles = fourier.size();
								dt = 2 * M_PI / double(numEpicycles);
								paused = false;
							}
						}
					}
					break;
				case sf::Event::MouseMoved:
					if (leftBtnDown && userDrawingMode) {
						sf::Vector2i mousePos = sf::Mouse::getPosition(window);
						userDrawingCoords.push_back({mousePos.x, mousePos.y});
					}
					break;
				case sf::Event::MouseButtonReleased:
					if (event.mouseButton.button == sf::Mouse::Left) {
						if (leftBtnDown && userDrawingMode) {
							sf::Vector2i mousePos = sf::Mouse::getPosition(window);
							userDrawingCoords.push_back({mousePos.x, mousePos.y});
							leftBtnDown = false;
						}
					}
					break;
				case sf::Event::KeyPressed:
					switch (event.key.code) {
						case sf::Keyboard::Up: case sf::Keyboard::Down:
							if (fourier.empty()) continue;
							if (event.key.code == sf::Keyboard::Up) {
								numEpicycles = std::min(numEpicycles * 2, int(fourier.size()));
							} else {
								int pow2 = pow(2, int(log2(numEpicycles)));
								numEpicycles = pow2 == numEpicycles ? pow2 / 2 : pow2;
								numEpicycles = std::max(numEpicycles, 2);
							}
							if (numEpicycles == fourier.size()) drawLabel("No. epicycles = " + std::to_string(numEpicycles) + " (max)", 500);
							else drawLabel("No. epicycles = " + std::to_string(numEpicycles), 500);
							path.clear();
							time = 0.0;
							continue;
						case sf::Keyboard::P:
							fourier = computeFourierFromCoords(PI);
							break;
						case sf::Keyboard::G:
							fourier = computeFourierFromCoords(GUITAR);
							break;
						case sf::Keyboard::T:
							fourier = computeFourierFromCoords(T_REX);
							break;
						case sf::Keyboard::Space:
							paused = !paused;
							continue;
					}
					userDrawingMode = paused = false;
					path.clear();
					time = 0.0;
					numEpicycles = fourier.size();
					dt = 2 * M_PI / double(numEpicycles);
					break;
			}
		}

		if (paused && !userDrawingMode) continue;

		window.clear(sf::Color::Black);

		if (userDrawingMode) {
			sf::VertexArray pixels(sf::Points);
			for (const pair<int, int>& coords : userDrawingCoords)
				pixels.append(sf::Vertex(sf::Vector2f(coords.first, coords.second), sf::Color::Red));
			window.draw(pixels);
		} else {  // Draw Fourier result
			pair<double, double> epicycleFinalPos = epicycles(SIZE / 2.0, SIZE / 2.0, fourier, time);
			path.push_back({epicycleFinalPos.first, epicycleFinalPos.second});
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

		if (userDrawingMode) drawLabel("Draw something, or select a preset with P/G/T. Right-click to exit drawing mode.", 0);
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

	return 0;
}
