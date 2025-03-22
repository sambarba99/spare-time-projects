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

int max_iters = ORIGINAL_MAX_ITERS;
double scale = ORIGINAL_SCALE;
double x_axis = WIDTH / 2;
double x_offset = x_axis;
double y_axis = HEIGHT / 2;
double y_offset = y_axis;
bool show_axes = true;
sf::Vertex x_axis_line[] = {
	sf::Vertex(sf::Vector2f(x_axis - 10, y_axis + LABEL_HEIGHT)),
	sf::Vertex(sf::Vector2f(x_axis + 10, y_axis + LABEL_HEIGHT))
};
sf::Vertex y_axis_line[] = {
	sf::Vertex(sf::Vector2f(x_axis, y_axis - 10 + LABEL_HEIGHT)),
	sf::Vertex(sf::Vector2f(x_axis, y_axis + 10 + LABEL_HEIGHT))
};
sf::RenderWindow window(
	sf::VideoMode(WIDTH, HEIGHT + LABEL_HEIGHT),
	"Click: set origin | 2/5/1/0: magnify by 2/5/10/100x | S: screenshot | T: toggle axes | Z/X: change max_iters | R: reset",
	sf::Style::Close
);
sf::Font font;


vector<int> linear_interpolate(const vector<int>& rgb1, const vector<int>& rgb2, const double t) {
	int new_r = rgb1[0] + t * (rgb2[0] - rgb1[0]);
	int new_g = rgb1[1] + t * (rgb2[1] - rgb1[1]);
	int new_b = rgb1[2] + t * (rgb2[2] - rgb1[2]);
	return {new_r, new_g, new_b};
}


sf::VertexArray get_pixels(const std::optional<complex<double>>& c_value = std::nullopt) {
	sf::VertexArray pixels(sf::Points, WIDTH * HEIGHT);

	int i;
	double real, imag, nu, t;
	vector<int> rgb1, rgb2, rgb;

	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			real = (double(x) - x_offset) / scale;  // x represents the real axis
			imag = (double(y) - y_offset) / scale;  // y represents the imaginary axis
			complex<double> z, c;

			if (RENDER_MANDELBROT && c_value == std::nullopt) {
				// z is fixed (0 + 0i), and c is being varied and tested
				c = {real, imag};
			} else {
				// c is fixed, and z is being varied and tested
				z = {real, imag};
				c = c_value.value_or(C_JULIA);
			}
			i = 0;
			while (abs(z) < BAILOUT_RADIUS && i < max_iters) {
				z = z * z + c;
				i++;
			}

			if (i < max_iters) {
				// Apply smooth colouring
				nu = i + 1 - log2(log2(abs(z)));
				nu /= max_iters;  // Normalise
				nu *= RGB_PALETTE.size();  // Scale
				t = nu - int(nu);  // Fractional part
				rgb1 = RGB_PALETTE[int(nu) % RGB_PALETTE.size()];
				rgb2 = RGB_PALETTE[int(nu + 1) % RGB_PALETTE.size()];
				rgb = linear_interpolate(rgb1, rgb2, t);
				pixels[y * WIDTH + x] = sf::Vertex(sf::Vector2f(x, y + LABEL_HEIGHT + 1), sf::Color(rgb[0], rgb[1], rgb[2]));
			}
		}
	}

	return pixels;
}


void draw_label(const std::string label_text) {
	sf::RectangleShape lbl_area(sf::Vector2f(WIDTH, LABEL_HEIGHT));
	lbl_area.setPosition(0, 0);
	lbl_area.setFillColor(sf::Color::Black);
	window.draw(lbl_area);

	sf::Text text(label_text, font, 14);
	sf::FloatRect text_rect = text.getLocalBounds();
	text.setOrigin(int(text_rect.left + text_rect.width / 2), int(text_rect.top + text_rect.height / 2));
	text.setPosition(WIDTH / 2, LABEL_HEIGHT / 2);
	text.setFillColor(sf::Color::White);
	window.draw(text);
	window.display();
}


void draw() {
	window.clear();
	window.draw(get_pixels());

	if (show_axes) {
		window.draw(x_axis_line, 2, sf::Lines);
		window.draw(y_axis_line, 2, sf::Lines);
	}

	double z_real = (WIDTH / 2 - x_offset) / scale;
	double z_imag = -(HEIGHT / 2 - y_offset) / scale;
	std::ostringstream z_real_str;
	std::ostringstream z_imag_str;
	std::ostringstream scale_str;
	z_real_str << std::setprecision(15) << z_real;
	z_imag_str << std::setprecision(15) << z_imag;
	scale_str << std::scientific << std::setprecision(4) << (scale / ORIGINAL_SCALE);
	draw_label("Current coords: (" + z_real_str.str() + ", " + z_imag_str.str() + ") | Current scale: " + scale_str.str());
}


void centre_around_origin() {
	x_offset -= x_axis - WIDTH / 2;
	y_offset -= y_axis - HEIGHT / 2 - LABEL_HEIGHT;
	x_axis = WIDTH / 2;
	y_axis = HEIGHT / 2;
}


void magnify(const double factor) {
	scale *= factor;
	x_offset = factor * (x_offset - x_axis) + x_axis;
	y_offset = factor * (y_offset - y_axis) + y_axis;
}


void plot_mandelbrot_zoom(const complex<double> c, const int num_steps, const double final_scale_factor, const int min_pixel_iters, const int max_pixel_iters) {
	// Given a complex point c, zoom in 'num_steps' times into this point such that the final scale is 'final_scale_factor'

	// Centre around the zoom target (given that we're starting at the origin (0,0),
	// dx and dy are just the real and imaginary components)
	double delta_x = c.real();
	double delta_y = -c.imag();  // y axis normally increases upwards, but does so downwards on a display
	x_offset -= delta_x * scale;
	y_offset -= delta_y * scale + LABEL_HEIGHT;
	centre_around_origin();

	// Scaling by step_scale_factor, num_steps times, will give us a magnification of final_scale_factor
	double step_scale_factor = pow(final_scale_factor, 1.0 / num_steps);

	double iter_step = double(max_pixel_iters - min_pixel_iters) / num_steps;
	vector<int> pixel_iters(num_steps + 1);
	for (int i = 0; i <= num_steps; i++)
		pixel_iters[i] = min_pixel_iters + i * iter_step;

	for (int i = 0; i <= num_steps; i++) {
		max_iters = pixel_iters[i];

		sf::Image image;
		image.create(WIDTH, HEIGHT);
		sf::VertexArray pixels = get_pixels();
		for (int i = 0; i < pixels.getVertexCount(); i++) {
			sf::Vector2f pos = pixels[i].position;
			sf::Color colour = pixels[i].color;
			if (pos.y >= LABEL_HEIGHT)  // Don't care about rendering the label
				image.setPixel(static_cast<unsigned int>(pos.x), static_cast<unsigned int>(pos.y - LABEL_HEIGHT - 1), colour);
		}
		std::ostringstream file_path;
		file_path << "C:/Users/sam/Desktop/frames/" << std::setw(4) << std::setfill('0') << i << ".png";
		image.saveToFile(file_path.str());

		if (i == num_steps) break;

		magnify(step_scale_factor);
	}
}


void plot_julia_rotation(const double r, const int num_steps) {
	// Given a radius r, generate 'num_steps' complex points using the formula: r x e^(ai)
	// (where 'a' is varied from 0 to 2 pi) and plot the Julia set for each point

	int screenshot_counter = 0;
	double dt = 2 * M_PI / num_steps;

	for (double a = 0.0; a <= 2 * M_PI; a += dt) {
		sf::Image image;
		image.create(WIDTH, HEIGHT);
		complex<double> c = r * complex<double>(cos(a), sin(a));
		sf::VertexArray pixels = get_pixels(c);
		for (int i = 0; i < pixels.getVertexCount(); i++) {
			sf::Vector2f pos = pixels[i].position;
			sf::Color colour = pixels[i].color;
			if (pos.y >= LABEL_HEIGHT)  // Don't care about rendering the label
				image.setPixel(static_cast<unsigned int>(pos.x), static_cast<unsigned int>(pos.y - LABEL_HEIGHT - 1), colour);
		}
		std::ostringstream file_path;
		file_path << "C:/Users/sam/Desktop/frames/" << std::setw(4) << std::setfill('0') << screenshot_counter << ".png";
		image.saveToFile(file_path.str());
		screenshot_counter++;
	}
}


int main() {
	// README .webp files created using these
	// plot_mandelbrot_zoom(complex<double>(-0.74453952, 0.12172412), 450, 5e4, 50, 1200);
	// plot_mandelbrot_zoom(complex<double>(0.360147036, 0.641212176), 600, 1e6, 50, 300);
	// plot_mandelbrot_zoom(complex<double>(-1.479892325756, 0.00063343092), 900, 1e9, 50, 2000);
	// plot_mandelbrot_zoom(complex<double>(-0.77468056281905, -0.13741669895407), 900, 1e12, 50, 1600);
	// plot_julia_rotation(0.77, 600);

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
						sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
						int mouse_x = mouse_pos.x, mouse_y = mouse_pos.y;
						if (mouse_y > LABEL_HEIGHT) {
							draw_label("Setting origin...");
							x_axis = mouse_x;
							y_axis = mouse_y;
							centre_around_origin();
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

							draw_label("Magnifying by " + to_string(int(factor)) + "x...");
							magnify(factor);
							draw();
							break;
						case sf::Keyboard::R:
							draw_label("Resetting...");
							max_iters = ORIGINAL_MAX_ITERS;
							scale = ORIGINAL_SCALE;
							x_axis = x_offset = WIDTH / 2;
							y_axis = y_offset = HEIGHT / 2;
							show_axes = true;
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
							draw_label("Toggling axes...");
							show_axes = !show_axes;
							draw();
							break;
						case sf::Keyboard::X:
							if (max_iters < ITER_LIMIT_MAX) {
								draw_label("Doubling max_iters...");
								max_iters *= 2;
								if (max_iters > ITER_LIMIT_MAX)
									max_iters = ITER_LIMIT_MAX;
								draw();
								std::cout << "max_iters = " << max_iters << '\n';
							}
							break;
						case sf::Keyboard::Z:
							if (max_iters > ITER_LIMIT_MIN) {
								draw_label("Halving max_iters...");
								max_iters /= 2;
								if (max_iters < ITER_LIMIT_MIN)
									max_iters = ITER_LIMIT_MIN;
								draw();
								std::cout << "max_iters = " << max_iters << '\n';
							}
							break;
					}
					break;
			}
		}
	}

	return 0;
}
