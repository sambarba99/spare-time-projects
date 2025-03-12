/*
Mandelbrot/Julia set visualiser

Controls:
	Click: select point to set as origin (0,0)
	Num keys 2,4,8,0: magnify around origin by 2/4/8/100 times, respectively
	S: screenshot
	T: toggle axes
	Z/X: change max iterations per pixel (resolution)
	R: reset

Author: Sam Barba
Created 18/11/2022
*/

#include <complex>
#include <iomanip>
#include <iostream>
#include <optional>
#include <SFML/Graphics.hpp>

using std::complex;
using std::to_string;
using std::vector;


const int WIDTH = 1200;
const int HEIGHT = 750;
const double ORIGINAL_SCALE = 300.0;
const double BAILOUT_RADIUS = 128.0;
const int ORIGINAL_MAX_ITERS = 200;
const int LABEL_HEIGHT = 30;
const vector<vector<int>> RGB_PALETTE = {{0, 0, 90}, {20, 60, 170}, {70, 160, 230}, {230, 255, 255}, {255, 200, 50}, {140, 60, 30}, {50, 0, 50}};

// Set to true if rendering the Mandelbrot set...
const bool RENDER_MANDELBROT = true;

// ...or, change this complex number C_JULIA
// const complex<double> C_JULIA(-0.8, 0.16);
// const complex<double> C_JULIA(-0.75, 0.11);
// const complex<double> C_JULIA(-0.4, 0.595);
// const complex<double> C_JULIA(-0.25, 0.76);
// const complex<double> C_JULIA(0.0, 0.7);
const complex<double> C_JULIA(0.28, 0.008);

int maxIters = ORIGINAL_MAX_ITERS;
double scale = ORIGINAL_SCALE;
double xAxis = WIDTH / 2;
double xOffset = xAxis;
double yAxis = HEIGHT / 2;
double yOffset = yAxis;
bool showAxes = true;
sf::Vertex xAxisLine[] = {
	sf::Vertex(sf::Vector2f(xAxis - 10, yAxis + LABEL_HEIGHT)),
	sf::Vertex(sf::Vector2f(xAxis + 10, yAxis + LABEL_HEIGHT))
};
sf::Vertex yAxisLine[] = {
	sf::Vertex(sf::Vector2f(xAxis, yAxis - 10 + LABEL_HEIGHT)),
	sf::Vertex(sf::Vector2f(xAxis, yAxis + 10 + LABEL_HEIGHT))
};
sf::RenderWindow window(
	sf::VideoMode(WIDTH, HEIGHT + LABEL_HEIGHT),
	"Click: set origin | 2/4/8/0: magnify by 2/4/8/100x | S: screenshot | T: toggle axes | Z/X: change maxIters | R: reset",
	sf::Style::Close
);
sf::Font font;


vector<int> linearInterpolate(const vector<int>& colour1, const vector<int>& colour2, const double t) {
	int newR = colour1[0] + t * (colour2[0] - colour1[0]);
	int newG = colour1[1] + t * (colour2[1] - colour1[1]);
	int newB = colour1[2] + t * (colour2[2] - colour1[2]);
	return {newR, newG, newB};
}


sf::VertexArray getPixels(const std::optional<complex<double>> cValue = std::nullopt) {
	sf::VertexArray pixels(sf::Points, WIDTH * HEIGHT);

	int i;
	double real, imag, nu, t;
	vector<int> colour1, colour2, colour;

	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			real = (double(x) - xOffset) / scale;  // x represents the real axis
			imag = (double(y) - yOffset) / scale;  // y represents the imaginary axis
			complex<double> z, c;

			if (RENDER_MANDELBROT && cValue == std::nullopt) {
				// z is fixed (0 + 0i), and c is being varied and tested
				c = {real, imag};
			} else {
				// c is fixed, and z is being varied and tested
				z = {real, imag};
				c = cValue.value_or(C_JULIA);
			}
			i = 0;
			while (abs(z) < BAILOUT_RADIUS && i < maxIters) {
				z = z * z + c;
				i++;
			}

			if (i < maxIters) {
				// Apply smooth colouring
				nu = i + 1 - log2(log2(abs(z)));
				nu /= maxIters;  // Normalise
				nu *= RGB_PALETTE.size();  // Scale
				t = nu - int(nu);  // Fractional part
				colour1 = RGB_PALETTE[int(nu) % RGB_PALETTE.size()];
				colour2 = RGB_PALETTE[int(nu + 1) % RGB_PALETTE.size()];
				colour = linearInterpolate(colour1, colour2, t);
				pixels[y * WIDTH + x] = sf::Vertex(sf::Vector2f(x, y + LABEL_HEIGHT), sf::Color(colour[0], colour[1], colour[2]));
			}
		}
	}

	return pixels;
}


void drawLabel(const std::string label) {
	sf::RectangleShape lblArea(sf::Vector2f(WIDTH, LABEL_HEIGHT));
	lblArea.setPosition(0, 0);
	lblArea.setFillColor(sf::Color::Black);
	window.draw(lblArea);

	sf::Text text(label, font, 14);
	sf::FloatRect textRect = text.getLocalBounds();
	text.setOrigin(int(textRect.left + textRect.width / 2), int(textRect.top + textRect.height / 2));
	text.setPosition(WIDTH / 2, LABEL_HEIGHT / 2);
	text.setFillColor(sf::Color::White);
	window.draw(text);
	window.display();
}


void draw() {
	window.clear(sf::Color::Black);
	window.draw(getPixels());

	if (showAxes) {
		window.draw(xAxisLine, 2, sf::Lines);
		window.draw(yAxisLine, 2, sf::Lines);
	}

	double zReal = (WIDTH / 2 - xOffset) / scale;
	double zImag = -(HEIGHT / 2 - yOffset) / scale;
	std::ostringstream scaleStr;
	scaleStr << std::scientific << std::setprecision(4) << (scale / ORIGINAL_SCALE);
	drawLabel("Current coords: (" + to_string(zReal) + ", " + to_string(zImag) + ") | Current scale: " + scaleStr.str());
}


void centreAroundOrigin() {
	xOffset -= xAxis - WIDTH / 2;
	yOffset -= yAxis - HEIGHT / 2 - LABEL_HEIGHT;
	xAxis = WIDTH / 2;
	yAxis = HEIGHT / 2;
}


void magnify(const double factor) {
	scale *= factor;
	xOffset = factor * (xOffset - xAxis) + xAxis;
	yOffset = factor * (yOffset - yAxis) + yAxis;
}


void plotMandelbrotZoom(const complex<double> c, const double scaleFactor, const int numSteps, const int pixelIters) {
	// Given a complex point c, zoom in 'numSteps' times about this point by amount 'scaleFactor'

	double zoomTargetX = c.real() * scale + WIDTH / 2;
	double zoomTargetY = -c.imag() * scale + HEIGHT / 2 + LABEL_HEIGHT;
	double deltaX = xAxis - zoomTargetX;
	double deltaY = yAxis - zoomTargetY;
	xOffset += deltaX;
	yOffset += deltaY;
	centreAroundOrigin();
	maxIters = 50;  // pixelIters;

	for (int i = 0; i <= numSteps; i++) {
		sf::Image image;
		image.create(WIDTH, HEIGHT);
		sf::VertexArray pixels = getPixels();
		for (int i = 0; i < pixels.getVertexCount(); i++) {
			sf::Vector2f pos = pixels[i].position;
			sf::Color colour = pixels[i].color;
			if (pos.y >= LABEL_HEIGHT)  // Don't care about rendering the label
				image.setPixel(static_cast<unsigned int>(pos.x), static_cast<unsigned int>(pos.y - LABEL_HEIGHT), colour);
		}
		std::ostringstream filePath;
		filePath << "C:/Users/sam/Desktop/frames/" << std::setw(6) << std::setfill('0') << i << ".png";
		image.saveToFile(filePath.str());

		// Shift axes to move into zoom target
		// xOffset += deltaX / numSteps;
		// yOffset += deltaY / numSteps;
		// centreAroundOrigin();
		magnify(scaleFactor);
	}

	// Reset to defaults
	maxIters = ORIGINAL_MAX_ITERS;
	scale = ORIGINAL_SCALE;
	xAxis = xOffset = WIDTH / 2;
	yAxis = yOffset = HEIGHT / 2;
}


void plotJuliaInterpolation(const complex<double> cStart, const complex<double> cEnd, const int numSteps) {
	// Given two complex points, cStart and cEnd, linearly interpolate between them (numSteps)
	// and plot the Julia set for each interpolated point

	int screenshotCounter = 0;
	double dt = 1.0 / numSteps;

	for (double t = 0.0; t <= 1.0; t += dt) {
		sf::Image image;
		image.create(WIDTH, HEIGHT);
		complex<double> c = (1 - t) * cStart + t * cEnd;
		sf::VertexArray pixels = getPixels(c);
		for (int i = 0; i < pixels.getVertexCount(); i++) {
			sf::Vector2f pos = pixels[i].position;
			sf::Color colour = pixels[i].color;
			if (pos.y >= LABEL_HEIGHT)  // Don't care about rendering the label
				image.setPixel(static_cast<unsigned int>(pos.x), static_cast<unsigned int>(pos.y - LABEL_HEIGHT), colour);
		}
		std::ostringstream filePath;
		filePath << "C:/Users/sam/Desktop/frames/" << std::setw(6) << std::setfill('0') << screenshotCounter << ".png";
		image.saveToFile(filePath.str());
		screenshotCounter++;
	}
}


void plotJuliaRotation(const double r, const int numSteps) {
	// Given a radius r, generate 'numSteps' complex points using the formula: r x e^(ai)
	// (where 'a' is varied from 0 to 2 pi) and plot the Julia set for each point

	int screenshotCounter = 0;
	double dt = 2 * M_PI / numSteps;

	for (double a = 0.0; a <= 2 * M_PI; a += dt) {
		sf::Image image;
		image.create(WIDTH, HEIGHT);
		complex<double> c = r * complex<double>(cos(a), sin(a));
		sf::VertexArray pixels = getPixels(c);
		for (int i = 0; i < pixels.getVertexCount(); i++) {
			sf::Vector2f pos = pixels[i].position;
			sf::Color colour = pixels[i].color;
			if (pos.y >= LABEL_HEIGHT)  // Don't care about rendering the label
				image.setPixel(static_cast<unsigned int>(pos.x), static_cast<unsigned int>(pos.y - LABEL_HEIGHT), colour);
		}
		std::ostringstream filePath;
		filePath << "C:/Users/sam/Desktop/frames/" << std::setw(6) << std::setfill('0') << screenshotCounter << ".png";
		image.saveToFile(filePath.str());
		screenshotCounter++;
	}
}

int main() {
	plotMandelbrotZoom(complex<double>(-0.777878, 0.135762), 1.866, 10, 200);
	// plotJuliaInterpolation(complex<double>(-0.8, 0.16), complex<double>(-0.75, 0.11), 300);
	// plotJuliaRotation(0.77, 500);

	font.loadFromFile("C:/Windows/Fonts/consola.ttf");
	double factor;
	sf::Texture texture;
	texture.create(window.getSize().x, window.getSize().y);
	sf::Image screenshot;
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
							if (event.key.code == sf::Keyboard::Num2) factor = 2.0;
							else if (event.key.code == sf::Keyboard::Num4) factor = 4.0;
							else if (event.key.code == sf::Keyboard::Num8) factor = 8.0;
							else factor = 100.0;

							drawLabel("Magnifying by " + to_string(int(factor)) + "x...");
							magnify(factor);
							draw();
							break;
						case sf::Keyboard::R:
							drawLabel("Resetting...");
							scale = ORIGINAL_SCALE;
							maxIters = ORIGINAL_MAX_ITERS;
							xAxis = xOffset = WIDTH / 2;
							yAxis = yOffset = HEIGHT / 2;
							showAxes = true;
							draw();
							break;
						case sf::Keyboard::S:
							window.display();
							texture.update(window);
							screenshot = texture.copyToImage();
							screenshot.saveToFile("./images/screenshot.png");
							std::cout << "Screenshot saved at ./images/screenshot.png\n";
							window.display();
							break;
						case sf::Keyboard::T:
							drawLabel("Toggling axes...");
							showAxes = !showAxes;
							draw();
							break;
						case sf::Keyboard::X:
							if (maxIters <= 800) {
								drawLabel("Doubling maxIters...");
								maxIters *= 2;
								draw();
								std::cout << "maxIters = " << maxIters << '\n';
							}
							break;
						case sf::Keyboard::Z:
							if (maxIters >= 100) {
								drawLabel("Halving maxIters...");
								maxIters /= 2;
								draw();
								std::cout << "maxIters = " << maxIters << '\n';
							}
							break;
					}
					break;
			}
		}
	}

	return 0;
}
