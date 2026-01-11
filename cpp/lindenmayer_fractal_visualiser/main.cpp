/*
Fractal drawing using Lindenmayer systems (L-systems)

Controls:
	Click to cycle through drawings

Author: Sam Barba
Created 15/11/2022
*/

#include <cmath>
#include <iomanip>
#include <regex>
#include <SFML/Graphics.hpp>
#include <unordered_map>

using std::string;
using std::vector;


struct Fractal {
	string name;
	string axiom;
	std::unordered_map<char, string> ruleset;
	int max_iters;
	int turn_angle;
	int start_heading;
};

/*
Below are rules that generate interesting geometry. Their alphabet:
	F,G: move forward
	+: turn clockwise
	-: turn anti-clockwise
	<: decrease step size
	[: save current state
	]: return to last saved state
	Other characters are ignored (simply placeholders).
*/

const Fractal BINARY_TREE = {"Binary tree", "F", {{'F', "G[-F]+F"}, {'G', "GG"}}, 8, 45, -90};
const Fractal H_FIGURE = {"'H' figure", "[F]--[F]", {{'F', "<G[+F][-F]"}}, 12, 90, 0};
const Fractal SIERPINSKI_TRIANGLE = {"Sierpinski triangle", "F-G-G", {{'F', "F-G+F+G-F"}, {'G', "GG"}}, 7, 120, 0};
const Fractal SIERPINSKI_ARROWHEAD = {"Sierpinski arrowhead", "F", {{'F', "G-F-G"}, {'G', "F+G+F"}}, 9, 60, 240};
const Fractal KOCH_SNOWFLAKE = {"Koch snowflake", "F--F--F", {{'F', "F+F--F+F"}}, 5, 60, 0};
const Fractal KOCH_ISLAND = {"Koch island", "F+F+F+F", {{'F', "F-F+F+FFF-F-F+F"}}, 4, 90, 0};
const Fractal KOCH_RING = {"Koch ring", "F+F+F+F", {{'F', "FF+F+F+F+F+F-F"}}, 5, 90, 0};
const Fractal PENTAPLEXITY = {"Pentaplexity", "F++F++F++F++F", {{'F', "F++F++F+++++F-F++F"}}, 5, 36, 36};
const Fractal TRIANGLES = {"Triangles", "F+F+F", {{'F', "F-F+F"}}, 8, 120, 0};
const Fractal PENROSE = {"Penrose", "[1]++[1]++[1]++[1]++[1]", {{'F', ""}, {'0', "2F++3F----1F[-2F----0F]++"}, {'1', "+2F--3F[---0F--1F]+"}, {'2', "-0F++1F[+++2F++3F]-"}, {'3', "--2F++++0F[+3F++++1F]--1F"}}, 7, 36, -90};
const Fractal PEANO_GOSPER_CURVE = {"Peano-Gosper curve", "F0", {{'0', "0+1F++1F-F0--F0F0-1F+"}, {'1', "-F0+1F1F++1F+F0--F0-1"}}, 5, 60, 180};
const Fractal HILBERT_CURVE = {"Hilbert curve", "0", {{'0', "+1F-0F0-F1+"}, {'1', "-0F+1F1+F0-"}}, 8, 90, 180};
const Fractal LEVY_C_CURVE = {"Levy C curve", "F", {{'F', "+F--F+"}}, 16, 45, 0};
const Fractal DRAGON_CURVE = {"Dragon curve", "F0", {{'F', ""}, {'0', "-F0++F1-"}, {'1', "+F0--F1+"}}, 16, 45, 0};
const Fractal ASYMMETRIC_TREE_1 = {"Asymmetric tree 1", "F", {{'F', "G+[[F]-F]-G[-GF]+F"}}, 7, 15, -90};
const Fractal ASYMMETRIC_TREE_2 = {"Asymmetric tree 2", "F", {{'F', "FF[++F][-FF]"}}, 7, 20, -90};
const Fractal ASYMMETRIC_TREE_3 = {"Asymmetric tree 3", "F", {{'F', "FF+[+F-F-F]-[-F+F+F]"}}, 5, 20, -90};

const int WIDTH = 1500;
const int HEIGHT = 900;

string name_label;
sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Drawing with L-systems", sf::Style::Close);
sf::Font font;
int screenshot_counter = 0;


string generate_instructions(const string& axiom, const std::unordered_map<char, string>& ruleset, const int n) {
	/*
	Generates instructions from a ruleset applied to an initial axiom
	E.g. Lévy C curve rule {'F': "+F--F+"} applied to axiom "F" 3 times:
	               F -> +F--F+
	          +F--F+ -> ++F--F+--+F--F++
	++F--F+--+F--F++ -> +++F--F+--+F--F++--++F--F+--+F--F+++
	*/

	string instructions = axiom, instructions_new;
	for (int i = 0; i < n; i++) {
		instructions_new = "";
		for (char c : instructions) {
			if (ruleset.count(c))
				instructions_new += ruleset.at(c);
			else
				instructions_new += c;
		}
		instructions = instructions_new;
	}

	// Remove consecutive commands that cancel out
	instructions = regex_replace(instructions, std::regex(R"(\+-)"), "");
	instructions = regex_replace(instructions, std::regex(R"(-\+)"), "");
	return instructions;
}


vector<vector<double>> scale_and_centre_coords(vector<vector<double>>& coords) {
	/*
	First calculate scale factor k: image must fill 85% of either screen's width or height,
	depending on if the image is wider than it is tall or vice-versa.
	*/
	double x_min = std::numeric_limits<double>::max(), x_max = std::numeric_limits<double>::min();
	double y_min = x_min, y_max = x_max;
	for (const auto& coord_set : coords) {
		if (coord_set[0] < x_min) x_min = coord_set[0];
		if (coord_set[2] < x_min) x_min = coord_set[2];
		if (coord_set[0] > x_max) x_max = coord_set[0];
		if (coord_set[2] > x_max) x_max = coord_set[2];
		if (coord_set[1] < y_min) y_min = coord_set[1];
		if (coord_set[3] < y_min) y_min = coord_set[3];
		if (coord_set[1] > y_max) y_max = coord_set[1];
		if (coord_set[3] > y_max) y_max = coord_set[3];
	}

	double kx = x_max > x_min ? (WIDTH * 0.85) / (x_max - x_min) : WIDTH * 0.85;
	double ky = y_max > y_min ? (HEIGHT * 0.85) / (y_max - y_min) : HEIGHT * 0.85;
	double k = std::min(kx, ky);

	for (int i = 0; i < coords.size(); i++) {
		coords[i][0] *= k;
		coords[i][1] *= k;
		coords[i][2] *= k;
		coords[i][3] *= k;
	}

	// Now centre image about (WIDTH / 2, HEIGHT / 2)

	double mean_x = k * (x_min + x_max) / 2.0;
	double mean_y = k * (y_min + y_max) / 2.0;

	for (int i = 0; i < coords.size(); i++) {
		coords[i][0] -= mean_x - WIDTH / 2.0;
		coords[i][2] -= mean_x - WIDTH / 2.0;
		coords[i][1] -= mean_y - HEIGHT / 2.0;
		coords[i][3] -= mean_y - HEIGHT / 2.0;
	}

	return coords;
}


vector<int> hsv2rgb(const float h, const float s, const float v) {
	/*
	HSV to RGB
	0 <= hue < 360
	0 <= saturation <= 1
	0 <= value <= 1
	*/

	float c = s * v;
	float x = c * (1 - std::abs(fmod(h / 60.0, 2) - 1));
	float m = v - c;

	float rf, gf, bf;
	if (h < 60) rf = c, gf = x, bf = 0;
	else if (h < 120) rf = x, gf = c, bf = 0;
	else if (h < 180) rf = 0, gf = c, bf = x;
	else if (h < 240) rf = 0, gf = x, bf = c;
	else if (h < 300) rf = x, gf = 0, bf = c;
	else rf = c, gf = 0, bf = x;

	int r = (rf + m) * 255;
	int g = (gf + m) * 255;
	int b = (bf + m) * 255;

	return {r, g, b};
}


bool execute_instructions(const string& instructions, const double start_heading, const int turn_angle) {
	// If there's no 'move forward' command, it means no drawing, so return
	if (instructions.find('F') == string::npos && instructions.find('G') == string::npos)
		return false;

	// State contains current x, y, heading, step size
	// Start at 0,0 with step size 1
	vector<double> state = {0.0, 0.0, start_heading, 1.0};
	std::stack<vector<double>> saved_states;
	vector<vector<double>> coords_to_draw;

	// Execute the 'program', one char (command) at a time
	double x, y, heading, step_size, next_x, next_y;
	for (char cmd : instructions) {
		x = state[0];
		y = state[1];
		heading = state[2];
		step_size = state[3];

		switch (cmd) {
			case 'F':  // Move forward
			case 'G':
				next_x = step_size * cos(heading * M_PI / 180.0) + x;
				next_y = step_size * sin(heading * M_PI / 180.0) + y;
				state = {next_x, next_y, heading, step_size};
				coords_to_draw.push_back({x, y, next_x, next_y});
				break;
			case '<':  // Decrease step size
				state = {x, y, heading, step_size * 0.67};
				break;
			case '+':  // Turn clockwise
				state = {x, y, heading + turn_angle, step_size};
				break;
			case '-':  // Turn anti-clockwise
				state = {x, y, heading - turn_angle, step_size};
				break;
			case '[':  // Save current state
				saved_states.push(state);
				break;
			case ']':  // Return to last saved state
				state = saved_states.top();
				saved_states.pop();
				break;
		}
	}

	scale_and_centre_coords(coords_to_draw);

	// Draw with hue increasing from red (hue 0) to yellow (60)
	window.clear();
	for (int i = 0; i < coords_to_draw.size(); i++) {
		vector<double> coord_set = coords_to_draw[i];
		int start_x = round(coord_set[0]), start_y = round(coord_set[1]);
		int end_x = round(coord_set[2]), end_y = round(coord_set[3]);
		float hue = float(i) / float(coords_to_draw.size()) * 60.f;
		vector<int> rgb = hsv2rgb(hue, 1.f, 1.f);
		sf::Vertex line[] = {
			sf::Vertex(sf::Vector2f(start_x, start_y), sf::Color(rgb[0], rgb[1], rgb[2])),
			sf::Vertex(sf::Vector2f(end_x, end_y), sf::Color(rgb[0], rgb[1], rgb[2]))
		};
		window.draw(line, 2, sf::Lines);
	}

	// Draw label with fractal name and iteration no.
	sf::RectangleShape lbl_area(sf::Vector2f(WIDTH, 50));
	lbl_area.setPosition(0, 0);
	lbl_area.setFillColor(sf::Color::Black);
	window.draw(lbl_area);

	sf::Text text(name_label, font, 16);
	sf::FloatRect text_rect = text.getLocalBounds();
	text.setOrigin(int(text_rect.left + text_rect.width / 2), int(text_rect.top + text_rect.height / 2));
	text.setPosition(WIDTH / 2, 25);
	text.setFillColor(sf::Color::White);
	window.draw(text);

	window.display();

	// sf::Texture texture;
	// sf::Image screenshot;
	// texture.create(window.getSize().x, window.getSize().y);
	// texture.update(window);
	// screenshot = texture.copyToImage();
	// std::ostringstream file_path;
	// file_path << "C:/Users/sam/Desktop/frames/" << std::setw(4) << std::setfill('0') << screenshot_counter << ".png";
	// screenshot.saveToFile(file_path.str());
	// screenshot_counter++;

	return true;
}


void await_click() {
	sf::Event event;

	while (true) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					throw std::exception();
				case sf::Event::MouseButtonPressed:
					return;
			}
		}
	}
}


int main() {
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");

	// Draw each fractal, each from iteration 0 to its max iteration
	vector<Fractal> all_fractals = {BINARY_TREE, H_FIGURE, SIERPINSKI_TRIANGLE, SIERPINSKI_ARROWHEAD, KOCH_SNOWFLAKE,
		KOCH_ISLAND, KOCH_RING, PENTAPLEXITY, TRIANGLES, PENROSE, PEANO_GOSPER_CURVE, HILBERT_CURVE, LEVY_C_CURVE,
		DRAGON_CURVE, ASYMMETRIC_TREE_1, ASYMMETRIC_TREE_2, ASYMMETRIC_TREE_3};

	for (const Fractal& fract : all_fractals) {
		for (int i = 0; i <= fract.max_iters; i++) {
			name_label = fract.name + " (iteration " + std::to_string(i) + '/' + std::to_string(fract.max_iters) + ')';
			string instructions = generate_instructions(fract.axiom, fract.ruleset, i);
			bool drawing_done = execute_instructions(instructions, double(fract.start_heading), fract.turn_angle);
			if (drawing_done)
				await_click();
		}
	}

	window.close();

	return 0;
}
