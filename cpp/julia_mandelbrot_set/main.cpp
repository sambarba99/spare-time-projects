/*
Julia/Mandelbrot set visualiser

Controls:
	Click: select point to set as origin (0,0)
	Num keys 2,4,8,0: magnify around origin by 2/4/8/100 times, respectively
	T: toggle axes
	R: reset

Author: Sam Barba
Created 18/11/2022
*/

#include <complex>
#include <iomanip>
#include <SFML/Graphics.hpp>

using std::complex;
using std::to_string;
using std::vector;


const double WIDTH = 1200;
const double HEIGHT = 750;
const double ORIGINAL_SCALE = 300.0;
const int MAX_ITER = 200;
const int LABEL_HEIGHT = 25;
const vector<vector<int>> RGB_PALETTE = {{0, 20, 100}, {30, 100, 200}, {230, 255, 255}, {255, 170, 0}};

// Set to true if drawing Mandelbrot set...
const bool drawingMandelbrot = true;

/*
...or, change this complex number c: for a given c, its Julia set is the set of all complex z
for which the iteration z = z^2 + c doesn't diverge. For almost all c, these sets are fractals.
Interesting values of c:
*/
// complex<double> c(-0.79, 0.15);
// complex<double> c(-0.75, 0.11);
// complex<double> c(-0.7, 0.38);
// complex<double> c(-0.4, 0.595);
complex<double> c(0.28, 0.008);

double scale = ORIGINAL_SCALE;
double xAxis = WIDTH / 2;
double xOffset = xAxis;
double yAxis = HEIGHT / 2;
double yOffset = yAxis;
bool showAxes = true;
sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT + LABEL_HEIGHT), "Julia/Mandelbrot set visualiser | Click: set origin | 2/4/8/0: magnify by 2/4/8/100x | T: toggle axes | R: reset", sf::Style::Close);
sf::Font font;


vector<int> linearInterpolate(const vector<int>& colour1, const vector<int>& colour2, const double t) {
	int newR = colour1[0] + t * (colour2[0] - colour1[0]);
	int newG = colour1[1] + t * (colour2[1] - colour1[1]);
	int newB = colour1[2] + t * (colour2[2] - colour1[2]);
	return {newR, newG, newB};
}


void drawLabel(const std::string label) {
	sf::RectangleShape lblArea(sf::Vector2f(WIDTH, LABEL_HEIGHT));
	lblArea.setPosition(0, HEIGHT);
	lblArea.setFillColor(sf::Color::Black);
	window.draw(lblArea);

	sf::Text text(label, font, 14);
	sf::FloatRect textRect = text.getLocalBounds();
	text.setOrigin(int(textRect.left + textRect.width / 2), int(textRect.top + textRect.height / 2));
	text.setPosition(WIDTH / 2, HEIGHT + LABEL_HEIGHT / 2);
	text.setFillColor(sf::Color::White);
	window.draw(text);
	window.display();
}


void draw() {
	window.clear(sf::Color::Black);
	sf::VertexArray pixels(sf::Points);

	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			double zReal = (double(x) - xOffset) / scale;
			double zImag = (double(y) - yOffset) / scale;
			complex<double> z(zReal, zImag);

			if (drawingMandelbrot) c = z;

			// Test, as we iterate z = z^2 + c, does z diverge?
			int i = 0;
			while (abs(z) < 2.0 && i < MAX_ITER) {
				z = z * z + c;
				i++;
			}

			if (i < MAX_ITER) {
				// Apply smooth colouring
				double mu = (abs(z) > 1) ? (i + 1 - log2(log2(abs(z)))) : i;
				double muNorm = mu / (MAX_ITER + log2(log2(2.0)));
				double muScaled = muNorm * (RGB_PALETTE.size() - 1);
				int idx1 = int(muScaled);
				int idx2 = idx1 + 1;
				double fracPart = muScaled - idx1;
				vector<int> colour = linearInterpolate(RGB_PALETTE[idx1], RGB_PALETTE[idx2], fracPart);
				pixels.append(sf::Vertex(sf::Vector2f(x, y), sf::Color(colour[0], colour[1], colour[2])));
			}
		}
	}

	window.draw(pixels);

	if (showAxes) {
		sf::Vertex xAxisLine[] = {
			sf::Vertex(sf::Vector2f(0, yAxis)),
			sf::Vertex(sf::Vector2f(WIDTH, yAxis))
		};
		sf::Vertex yAxisLine[] = {
			sf::Vertex(sf::Vector2f(xAxis, 0)),
			sf::Vertex(sf::Vector2f(xAxis, HEIGHT))
		};
		window.draw(xAxisLine, 2, sf::Lines);
		window.draw(yAxisLine, 2, sf::Lines);
	}

	double zReal = (WIDTH / 2 - xOffset) / scale;
	double zImaginary = -(HEIGHT / 2 - yOffset) / scale;
	std::ostringstream zoomStr;
	zoomStr << std::scientific << std::setprecision(4) << (scale / ORIGINAL_SCALE);
	drawLabel("Current coords: (" + to_string(zReal) + ", " + to_string(zImaginary) + ") | Current zoom: " + zoomStr.str());
}


void centreAroundOrigin() {
	xOffset -= xAxis - WIDTH / 2;
	yOffset -= yAxis - HEIGHT / 2;
	xAxis = WIDTH / 2;
	yAxis = HEIGHT / 2;
}


void magnify(const double factor) {
	scale *= factor;
	xOffset = factor * (xOffset - xAxis) + xAxis;
	yOffset = factor * (yOffset - yAxis) + yAxis;
}


int main() {
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");
	double factor;
	sf::Event event;

	draw();

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::MouseButtonPressed:
					if (event.mouseButton.button == sf::Mouse::Left) {
						drawLabel("Setting origin...");
						sf::Vector2i mousePos = sf::Mouse::getPosition(window);
						xAxis = mousePos.x;
						yAxis = mousePos.y;
						centreAroundOrigin();
						draw();
					}
					break;
				case sf::Event::KeyPressed:
					switch (event.key.code) {
						case sf::Keyboard::Num2:
						case sf::Keyboard::Num4:
						case sf::Keyboard::Num8:
						case sf::Keyboard::Num0:
							factor = 2.0;
							if (event.key.code == sf::Keyboard::Num4) factor = 4.0;
							if (event.key.code == sf::Keyboard::Num8) factor = 8.0;
							if (event.key.code == sf::Keyboard::Num0) factor = 100.0;
							drawLabel("Magnifying by " + to_string(int(factor)) + "x...");
							magnify(factor);
							draw();
							break;
						case sf::Keyboard::T:
							drawLabel("Toggling axes...");
							showAxes = !showAxes;
							draw();
							break;
						case sf::Keyboard::R:
							drawLabel("Resetting...");
							scale = ORIGINAL_SCALE;
							xAxis = xOffset = WIDTH / 2;
							yAxis = yOffset = HEIGHT / 2;
							showAxes = true;
							draw();
							break;
					}
					break;
			}
		}
	}
}
