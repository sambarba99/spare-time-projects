/*
Mandelbrot/Julia set visualiser

Controls:
	Drag: move around
	Scroll: zoom in/out
	Z/X: change max iterations per pixel (resolution)
	A: toggle axes
	S: screenshot
	R: reset

Author: Sam Barba
Created 18/11/2022
*/

#include <complex>
#include <iomanip>
#include <iostream>
#include <optional>
#include <SFML/Graphics.hpp>
#include <thread>

using std::complex;


const int WIDTH = 1200;
const int HEIGHT = 750;
const int ORIGINAL_MAX_ITERS = 200;
const int ITER_LIMIT_MIN = 100;
const int ITER_LIMIT_MAX = 6400;
const int LABEL_HEIGHT = 30;
const double ORIGINAL_SCALE = 250.0;
const double BAILOUT_RADIUS = 128.0;
const double BAILOUT_RADIUS2 = BAILOUT_RADIUS * BAILOUT_RADIUS;
const double ZOOM_FACTOR = 2.0;
const std::vector<sf::Color> RGB_PALETTE = {
	{0, 0, 90}, {20, 60, 170}, {70, 160, 230}, {230, 255, 255}, {255, 200, 50}, {140, 60, 30}, {50, 0, 50},
	{140, 60, 30}, {255, 200, 50}, {230, 255, 255}, {70, 160, 230}, {20, 60, 170}
};

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

sf::Vector2i view_offset;
sf::Vertex x_axis_line[] = {
	sf::Vertex(sf::Vector2f(x_axis - 12, y_axis)),
	sf::Vertex(sf::Vector2f(x_axis + 11, y_axis))
};
sf::Vertex y_axis_line[] = {
	sf::Vertex(sf::Vector2f(x_axis, y_axis - 11)),
	sf::Vertex(sf::Vector2f(x_axis, y_axis + 12))
};
sf::RenderWindow window(
	sf::VideoMode(WIDTH, HEIGHT + LABEL_HEIGHT),
	"Drag: move around  |  Scroll: zoom  |  Z/X: change max_iters  |  A: toggle axes  |  S: screenshot  |  R: reset",
	sf::Style::Close
);
sf::Font font;

// For holding the static Mandelbrot/Julia set
sf::Image fractal_image, temp;
sf::Texture fractal_texture;
sf::Sprite fractal_sprite;

// Zoom preview variables
bool zoom_preview_active = false;
double zoom_preview_scale = 1.0;
sf::RectangleShape zoom_preview;


sf::Color linear_interpolate(const sf::Color& rgb1, const sf::Color& rgb2, const double t) {
	uint8_t new_r = rgb1.r + t * (rgb2.r - rgb1.r);
	uint8_t new_g = rgb1.g + t * (rgb2.g - rgb1.g);
	uint8_t new_b = rgb1.b + t * (rgb2.b - rgb1.b);
	return {new_r, new_g, new_b};
}


void compute_region(
	const int x_start = 0,
	const int y_start = 0,
	const int w = WIDTH,
	const int h = HEIGHT,
	const std::optional<complex<double>>& c_value = std::nullopt
) {
	int i;
	double real, imag, nu, t;
	double zx, zy, zx2, zy2, cx, cy;
	sf::Color rgb1, rgb2, rgb;

	for (int x = x_start; x < x_start + w; x++) {
		if (x < 0 || x >= WIDTH)
			continue;

		for (int y = y_start; y < y_start + h; y++) {
			if (y < 0 || y >= HEIGHT)
				continue;

			real = (double(x) - x_offset) / scale;  // x represents the real axis
			imag = (double(y) - y_offset) / scale;  // y represents the imaginary axis

			if (RENDER_MANDELBROT && !c_value) {
				// z is fixed (0 + 0i), and c is being varied and tested
				zx = 0.0;
				zy = 0.0;
				cx = real;
				cy = imag;
			} else {
				// c is fixed, and z is being varied and tested
				zx = real;
				zy = imag;
				cx = c_value ? c_value->real() : C_JULIA.real();
				cy = c_value ? c_value->imag() : C_JULIA.imag();
			}

			zx2 = zx * zx;
			zy2 = zy * zy;

			i = 0;
			while (zx2 + zy2 < BAILOUT_RADIUS2 && i < max_iters) {
				zy = 2.0 * zx * zy + cy;
				zx = zx2 - zy2 + cx;
				zx2 = zx * zx;
				zy2 = zy * zy;
				i++;
			}

			if (i < max_iters) {
				// Apply smooth colouring
				nu = i + 1 - log2(log2(zx2 + zy2) * 0.5);  // = log2(log2(abs(z))) if using std::complex
				nu /= max_iters;  // Normalise
				nu *= RGB_PALETTE.size();  // Scale
				t = nu - int(nu);  // Fractional part
				rgb1 = RGB_PALETTE[int(nu) % RGB_PALETTE.size()];
				rgb2 = RGB_PALETTE[int(nu + 1) % RGB_PALETTE.size()];
				rgb = linear_interpolate(rgb1, rgb2, t);
				fractal_image.setPixel(x, y, rgb);
			} else {
				fractal_image.setPixel(x, y, sf::Color::Black);
			}
		}
	}
}


void compute_region_threaded(
	int x_start = 0,
	int y_start = 0,
	int w = WIDTH,
	int h = HEIGHT,
	std::optional<std::complex<double>> c_value = std::nullopt
) {
	int num_threads = std::thread::hardware_concurrency();  // Use all available cores
	if (num_threads == 0)
		num_threads = 4;  // If detection fails

	std::vector<std::thread> threads;
	int block_height = h / num_threads;

	for (int i = 0; i < num_threads; i++) {
		int y0 = y_start + i * block_height;
		int h0 = (i == num_threads - 1) ? (y_start + h - y0) : block_height;  // Last thread takes the remainder

		threads.emplace_back(compute_region, x_start, y0, w, h0, c_value);
	}

	for (auto& t : threads)
		t.join();
}


void centre_around_origin() {
	x_offset -= x_axis - WIDTH / 2;
	y_offset -= y_axis - HEIGHT / 2;
	x_axis = WIDTH / 2;
	y_axis = HEIGHT / 2;
}


void magnify(const double mag_factor) {
	scale *= mag_factor;
	x_offset = mag_factor * (x_offset - x_axis) + x_axis;
	y_offset = mag_factor * (y_offset - y_axis) + y_axis;
}


void draw_label(const std::string label_text, const bool clear_window = true) {
	if (clear_window)
		window.clear();

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

	// Apply transformation during view dragging (also offset for label height)
	sf::RenderStates states1, states2;
	states1.transform.translate(view_offset.x, view_offset.y + LABEL_HEIGHT + 1);
	states2.transform.translate(0, LABEL_HEIGHT);

	fractal_texture.update(fractal_image);
	fractal_sprite.setTexture(fractal_texture);
	window.draw(fractal_sprite, states1);

	if (zoom_preview_active)
		window.draw(zoom_preview);

	if (show_axes) {
		window.draw(x_axis_line, 2, sf::Lines, states2);
		window.draw(y_axis_line, 2, sf::Lines, states2);
	}

	double z_real = (WIDTH / 2 - x_offset - view_offset.x) / scale;
	double z_imag = -(HEIGHT / 2 - y_offset - view_offset.y) / scale;
	std::ostringstream z_real_str, z_imag_str, scale_str;
	z_real_str << std::setprecision(15) << z_real;
	z_imag_str << std::setprecision(15) << z_imag;
	scale_str << std::scientific << std::setprecision(4) << (scale / ORIGINAL_SCALE);
	std::string re_str = z_real_str.str() == "-0" ? "0" : z_real_str.str();
	std::string im_str = z_imag_str.str() == "-0" ? "0" : z_imag_str.str();
	std::string label = "Coords: (" + re_str + ", " + im_str + ")  |  Scale: " + scale_str.str() + "  |  Zoom preview: ";
	std::string zoom_preview_scale_str = std::to_string(zoom_preview_scale > 1.0 ? int(zoom_preview_scale) : int(-1.0 / zoom_preview_scale)) + "x";
	label += zoom_preview_active ? zoom_preview_scale_str : "None";

	draw_label(label, false);
}


void plot_mandelbrot_zoom(const complex<double> c, const int num_steps, const double final_scale_factor, const int min_pixel_iters, const int max_pixel_iters) {
	// Given a complex point c, zoom in 'num_steps' times into this point such that the final scale is 'final_scale_factor'

	// Centre around the zoom target (given that we're starting at the origin (0,0),
	// dx and dy are just the real and imaginary components)
	double delta_x = c.real();
	double delta_y = -c.imag();  // y axis normally increases upwards, but does so downwards on a display
	x_offset -= delta_x * scale;
	y_offset -= delta_y * scale;
	centre_around_origin();

	// Scaling by step_scale_factor, num_steps times, will give us a magnification of final_scale_factor
	double step_scale_factor = std::pow(final_scale_factor, 1.0 / num_steps);

	double iter_step = double(max_pixel_iters - min_pixel_iters) / num_steps;
	std::vector<int> pixel_iters(num_steps + 1);
	for (int i = 0; i <= num_steps; i++)
		pixel_iters[i] = min_pixel_iters + i * iter_step;

	for (int i = 0; i <= num_steps; i++) {
		max_iters = pixel_iters[i];
		compute_region_threaded();
		std::ostringstream file_path;
		file_path << "C:/Users/sam/Desktop/frames/" << std::setw(4) << std::setfill('0') << i << ".png";
		fractal_image.saveToFile(file_path.str());

		if (i == num_steps)
			break;

		magnify(step_scale_factor);
	}
}


void plot_julia_rotation(const double r, const int num_steps) {
	// Given a radius r, generate 'num_steps' complex points using the formula: r x e^(ai)
	// (where 'a' is varied from 0 to 2 pi) and plot the Julia set for each point

	int screenshot_counter = 0;
	double dt = 2 * M_PI / num_steps;

	for (double a = 0.0; a <= 2 * M_PI; a += dt) {
		complex<double> c = r * complex<double>(cos(a), sin(a));
		compute_region_threaded(0, 0, WIDTH, HEIGHT, c);
		std::ostringstream file_path;
		file_path << "C:/Users/sam/Desktop/frames/" << std::setw(4) << std::setfill('0') << screenshot_counter << ".png";
		fractal_image.saveToFile(file_path.str());
		screenshot_counter++;
	}
}


int main() {
	fractal_image.create(WIDTH, HEIGHT, sf::Color::Black);
	fractal_texture.create(WIDTH, HEIGHT);

	// README .webp files created using these
	// plot_mandelbrot_zoom(complex<double>(-0.74453952, 0.12172412), 450, 5e4, 100, 2500);
	// plot_mandelbrot_zoom(complex<double>(0.360147036, 0.641212176), 600, 1e6, 100, 1000);
	// plot_mandelbrot_zoom(complex<double>(-1.479892325756, 0.00063343092), 900, 1e9, 100, 3000);
	// plot_mandelbrot_zoom(complex<double>(0.452360620009609, 0.396152988222595), 900, 1e12, 100, 4000);
	// plot_julia_rotation(0.77, 600);

	bool left_btn_down = false;
	sf::Vector2i drag_origin, mouse_pos;
	sf::Event event;
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");
	zoom_preview.setFillColor(sf::Color(255, 0, 128, 64));  // Translucent pink

	compute_region_threaded();
	draw();

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;

				case sf::Event::MouseButtonPressed:
					if (event.mouseButton.button == sf::Mouse::Left) {
						mouse_pos = sf::Mouse::getPosition(window);

						if (mouse_pos.y < LABEL_HEIGHT)
							continue;

						if (zoom_preview_active) {
							sf::Vector2f zoom_box_centre = zoom_preview.getPosition();

							// Convert screen coordinates to complex plane coordinates
							double real_centre = (zoom_box_centre.x - x_offset) / scale;
							double imag_centre = (zoom_box_centre.y - y_offset - LABEL_HEIGHT) / scale;

							// Update scale so the preview fills the window
							scale *= WIDTH / zoom_preview.getSize().x;

							// Apply new scale and adjust offsets so the preview centre stays in the same complex position
							x_offset = WIDTH / 2 - real_centre * scale;
							y_offset = HEIGHT / 2 - imag_centre * scale;

							// Reset zoom preview
							zoom_preview_active = false;
							zoom_preview_scale = 1.0;

							compute_region_threaded();
							draw();
						} else {
							left_btn_down = true;
							drag_origin = mouse_pos - view_offset;
						}
					}
					break;

				case sf::Event::MouseMoved:
					if (event.mouseMove.y < LABEL_HEIGHT)
						continue;

					if (left_btn_down) {
						// Dragging view
						view_offset = sf::Mouse::getPosition(window) - drag_origin;
						draw();
					} else if (zoom_preview_active) {
						// Moving the zoom preview
						zoom_preview.setPosition(
							static_cast<float>(event.mouseMove.x),
							static_cast<float>(event.mouseMove.y)
						);
						draw();
					}
					break;

				case sf::Event::MouseButtonReleased:
					if (event.mouseButton.button == sf::Mouse::Left) {
						left_btn_down = false;

						// Amount the view has moved in pixels
						int dx = view_offset.x;
						int dy = view_offset.y;

						if (dx == 0 && dy == 0)
							continue;

						// Shift the current image by dx, dy
						temp = fractal_image;
						for (int x = 0; x < WIDTH; x++) {
							for (int y = 0; y < HEIGHT; y++) {
								int orig_x = x - dx;
								int orig_y = y - dy;

								if (orig_x >= 0 && orig_x < WIDTH && orig_y >= 0 && orig_y < HEIGHT)
									fractal_image.setPixel(x, y, temp.getPixel(orig_x, orig_y));
								else
									fractal_image.setPixel(x, y, sf::Color::Black);
							}
						}

						// Now compute the newly appeared regions
						x_offset += dx;
						y_offset += dy;

						if (dx > 0)
							compute_region_threaded(0, 0, dx, HEIGHT);
						else if (dx < 0)
							compute_region_threaded(WIDTH + dx, 0, -dx, HEIGHT);

						if (dy > 0)
							compute_region_threaded(0, 0, WIDTH, dy);
						else if (dy < 0)
							compute_region_threaded(0, HEIGHT + dy, WIDTH, -dy);

						view_offset = {0, 0};
						draw();
					}
					break;

				case sf::Event::MouseWheelScrolled:
					if (event.mouseWheelScroll.delta > 0) {
						// Scroll up = zoom in
						zoom_preview_scale = std::min(zoom_preview_scale * ZOOM_FACTOR, 64.0);
					} else if (event.mouseWheelScroll.delta < 0) {
						// Scroll down = zoom out
						zoom_preview_scale = std::max(zoom_preview_scale / ZOOM_FACTOR, 1.0 / 64.0);
					}

					zoom_preview_active = zoom_preview_scale != 1.0;
					if (zoom_preview_active) {
						mouse_pos = sf::Mouse::getPosition(window);
						double preview_w = WIDTH / zoom_preview_scale;
						double preview_h = HEIGHT / zoom_preview_scale;
						zoom_preview.setSize(sf::Vector2f(preview_w, preview_h));
						zoom_preview.setOrigin(preview_w / 2, preview_h / 2);
						zoom_preview.setPosition(
							static_cast<float>(mouse_pos.x),
							static_cast<float>(mouse_pos.y)
						);
					}
					draw();
					break;

				case sf::Event::KeyPressed:
					switch (event.key.code) {
						case sf::Keyboard::Z:
							if (max_iters > ITER_LIMIT_MIN) {
								draw_label("Halving max_iters...");
								max_iters = std::max(max_iters / 2, ITER_LIMIT_MIN);
								compute_region_threaded();
								draw();
								std::cout << "max_iters = " << max_iters << '\n';
							}
							break;
						case sf::Keyboard::X:
							if (max_iters < ITER_LIMIT_MAX) {
								draw_label("Doubling max_iters...");
								max_iters = std::min(max_iters * 2, ITER_LIMIT_MAX);
								compute_region_threaded();
								draw();
								std::cout << "max_iters = " << max_iters << '\n';
							}
							break;
						case sf::Keyboard::A:
							show_axes = !show_axes;
							draw();
							break;
						case sf::Keyboard::S:
							fractal_image.saveToFile("./images/screenshot.png");
							std::cout << "Screenshot saved at ./images/screenshot.png\n";
							break;
						case sf::Keyboard::R:
							max_iters = ORIGINAL_MAX_ITERS;
							scale = ORIGINAL_SCALE;
							x_axis = x_offset = WIDTH / 2;
							y_axis = y_offset = HEIGHT / 2;
							show_axes = true;
							zoom_preview_active = false;
							zoom_preview_scale = 1.0;
							view_offset = {0, 0};
							compute_region_threaded();
							draw();
							break;
					}
					break;
			}
		}
	}

	return 0;
}
