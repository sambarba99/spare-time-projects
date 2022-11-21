/*
Fractal drawing using Lindenmayer systems (L-systems)

Author: Sam Barba
Created 15/11/2022

Controls:
Click to cycle through drawings
*/

#include <algorithm>
#include <cmath>
#include <bits/stdc++.h>
#include <exception>
#include <map>
#include <regex>
#include <SFML/Graphics.hpp>
#include <stack>
#include <string>
#include <vector>

using std::exception;
using std::map;
using std::min;
using std::numeric_limits;
using std::regex;
using std::regex_replace;
using std::stack;
using std::string;
using std::to_string;
using std::vector;

struct Fractal {
	string name;
	string axiom;
	map<char, string> ruleset;
	int maxIters;
	int turnAngle;
	int startHeading;
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
const Fractal SIERPINSKI_ARROWHEAD = {"Sierpinski arrowhead", "F", {{'F', "G-F-G"}, {'G', "F+G+F"}}, 7, 60, 240};
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

string nameLabel;
sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Drawing with L-systems", sf::Style::Close);

string generateInstructions(const string axiom, const map<char, string> ruleset, const int n) {
	/*
	Generates instructions from a ruleset applied to an initial axiom
	E.g. Lévy C curve rule {'F': "+F--F+"} applied to axiom "F" 3 times:
	               F -> +F--F+
	          +F--F+ -> ++F--F+--+F--F++
	++F--F+--+F--F++ -> +++F--F+--+F--F++--++F--F+--+F--F+++
	*/

	string instructions = axiom, instructionsNew;
	for (int i = 0; i < n; i++) {
		instructionsNew = "";
		for (char c : instructions) {
			if (ruleset.count(c)) instructionsNew += ruleset.at(c);
			else instructionsNew += c;
		}
		instructions = instructionsNew;
	}

	// Remove command pairs that cancel each other
	instructions = regex_replace(instructions, regex(R"(\+-)"), "");
	instructions = regex_replace(instructions, regex(R"(-\+)"), "");
	return instructions;
}

vector<vector<double>> scaleAndCentreCoords(vector<vector<double>> coords) {
	/*
	First calculate scale factor k: image must fill 85% of either screen's width or height,
	depending on if the image is wider than it is tall or vice-versa.
	*/
	double xMin = numeric_limits<double>::max(), xMax = numeric_limits<double>::min();
	double yMin = xMin, yMax = xMax;
	for (vector<double> coordSet : coords) {
		if (coordSet[0] < xMin) xMin = coordSet[0];
		if (coordSet[2] < xMin) xMin = coordSet[2];
		if (coordSet[0] > xMax) xMax = coordSet[0];
		if (coordSet[2] > xMax) xMax = coordSet[2];
		if (coordSet[1] < yMin) yMin = coordSet[1];
		if (coordSet[3] < yMin) yMin = coordSet[3];
		if (coordSet[1] > yMax) yMax = coordSet[1];
		if (coordSet[3] > yMax) yMax = coordSet[3];
	}

	double kx = xMax > xMin ? (WIDTH * 0.85) / (xMax - xMin) : WIDTH * 0.85;
	double ky = yMax > yMin ? (HEIGHT * 0.85) / (yMax - yMin) : HEIGHT * 0.85;
	double k = min(kx, ky);

	for (int i = 0; i < coords.size(); i++) {
		coords[i][0] *= k;
		coords[i][1] *= k;
		coords[i][2] *= k;
		coords[i][3] *= k;
	}

	// Now centre image about (WIDTH / 2, HEIGHT / 2)

	double meanX = k * (xMin + xMax) / 2.0;
	double meanY = k * (yMin + yMax) / 2.0;

	for (int i = 0; i < coords.size(); i++) {
		coords[i][0] -= meanX - WIDTH / 2.0;
		coords[i][2] -= meanX - WIDTH / 2.0;
		coords[i][1] -= meanY - HEIGHT / 2.0;
		coords[i][3] -= meanY - HEIGHT / 2.0;
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
	float x = c * (1 - abs(fmod(h / 60.0, 2) - 1));
	float m = v - c;

	float rf, gf, bf;
	if (0 <= h && h < 60) rf = c, gf = x, bf = 0;
	else if (60 <= h && h < 120) rf = x, gf = c, bf = 0;
	else if (120 <= h && h < 180) rf = 0, gf = c, bf = x;
	else if (180 <= h && h < 240) rf = 0, gf = x, bf = c;
	else if (240 <= h && h < 300) rf = x, gf = 0, bf = c;
	else rf = c, gf = 0, bf = x;

	int r = (rf + m) * 255;
	int g = (gf + m) * 255;
	int b = (bf + m) * 255;

	return {r, g, b};
}

bool executeInstructions(const string instructions, const double startHeading, const int turnAngle) {
	// If there's no 'move forward' command, it means no drawing, so return
	if (instructions.find('F') == string::npos && instructions.find('G') == string::npos)
		return false;

	// State contains current x, y, heading, step size
	// Start at 0,0 with step size 1
	vector<double> state = {0.0, 0.0, startHeading, 1.0};
	stack<vector<double>> savedStates;
	vector<vector<double>> coordsToDraw;

	// Execute the 'program', one char (command) at a time
	double x, y, heading, stepSize, nextX, nextY;
	for (char cmd : instructions) {
		x = state[0];
		y = state[1];
		heading = state[2];
		stepSize = state[3];

		switch (cmd) {
			case 'F': case 'G':  // Move forward
				nextX = stepSize * cos(heading * M_PI / 180.0) + x;
				nextY = stepSize * sin(heading * M_PI / 180.0) + y;
				state = {nextX, nextY, heading, stepSize};
				coordsToDraw.push_back({x, y, nextX, nextY});
				break;
			case '<':  // Decrease step size
				state = {x, y, heading, stepSize * 0.67};
				break;
			case '+':  // Turn clockwise
				state = {x, y, heading + turnAngle, stepSize};
				break;
			case '-':  // Turn anti-clockwise
				state = {x, y, heading - turnAngle, stepSize};
				break;
			case '[':  // Save current state
				savedStates.push(state);
				break;
			case ']':  // Return to last saved state
				state = savedStates.top();
				savedStates.pop();
				break;
		}
	}

	coordsToDraw = scaleAndCentreCoords(coordsToDraw);

	// Draw with hue increasing from red (hue 0) to yellow (60)
	window.clear(sf::Color::Black);
	for (int i = 0; i < coordsToDraw.size(); i++) {
		vector<double> coordSet = coordsToDraw[i];
		int startX = round(coordSet[0]), startY = round(coordSet[1]);
		int endX = round(coordSet[2]), endY = round(coordSet[3]);
		float hue = float(i) / float(coordsToDraw.size()) * 60.f;
		vector<int> rgb = hsv2rgb(hue, 1.f, 1.f);
		sf::Vertex line[] = {
			sf::Vertex(sf::Vector2f(startX, startY), sf::Color(rgb[0], rgb[1], rgb[2])),
			sf::Vertex(sf::Vector2f(endX, endY), sf::Color(rgb[0], rgb[1], rgb[2]))
		};
		window.draw(line, 2, sf::Lines);
	}

	// Draw label with fractal name and iteration no.
	sf::RectangleShape lblArea(sf::Vector2f(WIDTH, 50));
	lblArea.setPosition(0, 0);
	lblArea.setFillColor(sf::Color::Black);
	window.draw(lblArea);

	sf::Font font;
	font.loadFromFile("C:\\Windows\\Fonts\\consola.ttf");
	sf::Text text(nameLabel, font, 16);
	sf::FloatRect textRect = text.getLocalBounds();
	text.setOrigin(int(textRect.left + textRect.width / 2), int(textRect.top + textRect.height / 2));
	text.setPosition(WIDTH / 2, 25);
	text.setFillColor(sf::Color::White);
	window.draw(text);

	window.display();

	return true;
}

void waitForClick() {
	while (true) {
		sf::Event event;
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					throw exception();
				case sf::Event::MouseButtonPressed:
					return;
			}
		}
	}
}

int main() {
	// Draw each fractal, each from iteration 0 to its max (i.e. computer won't crash) iteration
	vector<Fractal> allFractals = {BINARY_TREE, H_FIGURE, SIERPINSKI_TRIANGLE, SIERPINSKI_ARROWHEAD, KOCH_SNOWFLAKE,
		KOCH_ISLAND, KOCH_RING, PENTAPLEXITY, TRIANGLES, PENROSE, PEANO_GOSPER_CURVE, HILBERT_CURVE, LEVY_C_CURVE,
		DRAGON_CURVE, ASYMMETRIC_TREE_1, ASYMMETRIC_TREE_2, ASYMMETRIC_TREE_3};

	for (Fractal fract : allFractals) {
		for (int i = 0; i <= fract.maxIters; i++) {
			nameLabel = fract.name + " (iteration " + to_string(i) + '/' + to_string(fract.maxIters) + ')';
			string instructions = generateInstructions(fract.axiom, fract.ruleset, i);
			bool drawingDone = executeInstructions(instructions, double(fract.startHeading), fract.turnAngle);
			if (drawingDone) waitForClick();
		}
	}

	window.close();
}
