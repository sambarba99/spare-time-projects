/*
Mandelbrot/Julia set visualiser

Controls:
	Click: select point to set as origin (0,0)
	Num keys 2,5,1,0: magnify around origin by 2/5/10/100 times, respectively
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
const int ORIGINAL_MAX_ITERS = 200;
const int ITER_LIMIT_MIN = 50;
const int ITER_LIMIT_MAX = 3200;
const int LABEL_HEIGHT = 30;
const double ORIGINAL_SCALE = 250.0;
const double BAILOUT_RADIUS = 128.0;
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
	"Click: set origin | 2/5/1/0: magnify by 2/5/10/100x | S: screenshot | T: toggle axes | Z/X: change maxIters | R: reset",
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
				pixels[y * WIDTH + x] = sf::Vertex(sf::Vector2f(x, y + LABEL_HEIGHT + 1), sf::Color(colour[0], colour[1], colour[2]));
			}
		}
	}

	return pixels;
}


void drawLabel(const std::string labelText) {
	sf::RectangleShape lblArea(sf::Vector2f(WIDTH, LABEL_HEIGHT));
	lblArea.setPosition(0, 0);
	lblArea.setFillColor(sf::Color::Black);
	window.draw(lblArea);

	sf::Text text(labelText, font, 14);
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
	std::ostringstream zRealStr;
	std::ostringstream zImagStr;
	std::ostringstream scaleStr;
	zRealStr << std::setprecision(15) << zReal;
	zImagStr << std::setprecision(15) << zImag;
	scaleStr << std::scientific << std::setprecision(4) << (scale / ORIGINAL_SCALE);
	drawLabel("Current coords: (" + zRealStr.str() + ", " + zImagStr.str() + ") | Current scale: " + scaleStr.str());
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


void plotMandelbrotZoom(const complex<double> c, const int numSteps, const double finalScaleFactor, const int minPixelIters, const int maxPixelIters) {
	// Given a complex point c, zoom in 'numSteps' times into this point such that the final scale is 'finalScaleFactor'

	// Centre around the zoom target (given that we're starting at the origin (0,0),
	// dx and dy are just the real and imaginary components)
	double deltaX = c.real();
	double deltaY = -c.imag();  // y axis normally increases upwards, but does so downwards on a display
	xOffset -= deltaX * scale;
	yOffset -= deltaY * scale + LABEL_HEIGHT;
	centreAroundOrigin();

	// Scaling by stepScaleFactor, numSteps times, will give us a magnification of finalScaleFactor
	double stepScaleFactor = pow(finalScaleFactor, 1.0 / numSteps);

	double iterStep = double(maxPixelIters - minPixelIters) / numSteps;
	vector<int> pixelIters;
	for (int i = 0; i <= numSteps; i++)
		pixelIters.push_back(minPixelIters + i * iterStep);

	for (int i = 0; i <= numSteps; i++) {
		maxIters = pixelIters[i];

		sf::Image image;
		image.create(WIDTH, HEIGHT);
		sf::VertexArray pixels = getPixels();
		for (int i = 0; i < pixels.getVertexCount(); i++) {
			sf::Vector2f pos = pixels[i].position;
			sf::Color colour = pixels[i].color;
			if (pos.y >= LABEL_HEIGHT)  // Don't care about rendering the label
				image.setPixel(static_cast<unsigned int>(pos.x), static_cast<unsigned int>(pos.y - LABEL_HEIGHT - 1), colour);
		}
		std::ostringstream filePath;
		filePath << "C:/Users/sam/Desktop/frames/" << std::setw(4) << std::setfill('0') << i << ".png";
		image.saveToFile(filePath.str());

		if (i == numSteps) break;

		magnify(stepScaleFactor);
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
				image.setPixel(static_cast<unsigned int>(pos.x), static_cast<unsigned int>(pos.y - LABEL_HEIGHT - 1), colour);
		}
		std::ostringstream filePath;
		filePath << "C:/Users/sam/Desktop/frames/" << std::setw(4) << std::setfill('0') << screenshotCounter << ".png";
		image.saveToFile(filePath.str());
		screenshotCounter++;
	}
}


int main() {
	// README .webp files created using these
	// plotMandelbrotZoom(complex<double>(-0.74453952, 0.12172412), 450, 5e4, 50, 1200);
	// plotMandelbrotZoom(complex<double>(0.360147036, 0.641212176), 600, 1e6, 50, 300);
	// plotMandelbrotZoom(complex<double>(-1.479892325756, 0.00063343092), 900, 1e9, 50, 2000);
	// plotMandelbrotZoom(complex<double>(-0.77468056281905, -0.13741669895407), 900, 1e12, 50, 1600);
	// plotJuliaRotation(0.77, 600);

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
						sf::Vector2i mousePos = sf::Mouse::getPosition(window);
						int mouseX = mousePos.x, mouseY = mousePos.y;
						if (mouseY > LABEL_HEIGHT) {
							drawLabel("Setting origin...");
							xAxis = mouseX;
							yAxis = mouseY;
							centreAroundOrigin();
							draw();
						}
					}
					break;
				case sf::Event::KeyPressed:
					switch (event.key.code) {
						case sf::Keyboard::Num2:
						case sf::Keyboard::Num5:
						case sf::Keyboard::Num1:
						case sf::Keyboard::Num0:
							if (event.key.code == sf::Keyboard::Num2) factor = 2.0;
							else if (event.key.code == sf::Keyboard::Num5) factor = 5.0;
							else if (event.key.code == sf::Keyboard::Num1) factor = 10.0;
							else factor = 100.0;

							drawLabel("Magnifying by " + to_string(int(factor)) + "x...");
							magnify(factor);
							draw();
							break;
						case sf::Keyboard::R:
							drawLabel("Resetting...");
							maxIters = ORIGINAL_MAX_ITERS;
							scale = ORIGINAL_SCALE;
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
							if (maxIters < ITER_LIMIT_MAX) {
								drawLabel("Doubling maxIters...");
								maxIters *= 2;
								if (maxIters > ITER_LIMIT_MAX)
									maxIters = ITER_LIMIT_MAX;
								draw();
								std::cout << "maxIters = " << maxIters << '\n';
							}
							break;
						case sf::Keyboard::Z:
							if (maxIters > ITER_LIMIT_MIN) {
								drawLabel("Halving maxIters...");
								maxIters /= 2;
								if (maxIters < ITER_LIMIT_MIN)
									maxIters = ITER_LIMIT_MIN;
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
