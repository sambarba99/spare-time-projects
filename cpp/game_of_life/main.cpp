/*
Conway's Game of Life

Controls:
	P: select a preset pattern
	Z/X: cycle through patterns
	R: randomise cells
	S: show simulation info
	Up/down arrows: change delay between each step
	Space: play/pause
	Mouse drag: move around scene
	Scroll: zoom

Author: Sam Barba
Created 14/11/2022
*/

#include <iomanip>
#include <iostream>
#include <random>
#include <SFML/Graphics.hpp>
#include <unordered_map>
#include <unordered_set>

#include "utils.h"


const int GRID_WIDTH = 1440;  // Pixels
const int GRID_HEIGHT = 840;
const int LABEL_HEIGHT = 30;
const int CELL_SIZES[] = {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20};  // Pixels (all divisible into GRID_WIDTH and GRID_HEIGHT)
const int DELAYS[] = {0, 25, 100, 250};  // Millisecs

std::unordered_set<Cell, CellHash> cells;  // List of coords of live cells

// For computing next generation
std::unordered_map<Cell, int, CellHash> live_neighbour_counts;
std::unordered_set<Cell, CellHash> new_cells;
Cell neighbour_cell;

int cell_size_idx = 4;  // Start with 5px cell size
int delay_idx = 1;
int pattern_idx = -1;
int generation_num;
int num_removed_cells;
bool show_info = true;
sf::Vector2f view_offset;
std::random_device rd;
std::mt19937 gen(rd());
sf::RenderWindow window(sf::VideoMode(GRID_WIDTH, GRID_HEIGHT + LABEL_HEIGHT), "Game of Life", sf::Style::Close);
sf::Font font;


void randomise_live_cells() {
	std::vector<Cell> all_coords;
	cell_size_idx = std::max(cell_size_idx, 2);  // Make sure cell size it at least 3px, otherwise simulation is slow
	int grid_width_cells = GRID_WIDTH / CELL_SIZES[cell_size_idx];
	int grid_height_cells = GRID_HEIGHT / CELL_SIZES[cell_size_idx];
	for (int x = 0; x < grid_width_cells; x++)
		for (int y = 0; y < grid_height_cells; y++)
			all_coords.push_back({x, y});

	std::shuffle(all_coords.begin(), all_coords.end(), gen);
	cells.clear();
	int num_live_cells = static_cast<int>(grid_width_cells * grid_height_cells * 0.125);  // 1 in 8 chance of being alive
	for (int i = 0; i < num_live_cells; i++)
		cells.insert(all_coords[i]);

	pattern_idx = -1;
	generation_num = 0;
	num_removed_cells = 0;
}


void set_pattern(const Pattern& pattern) {
	cells.clear();
	int min_x = INT_MAX, max_x = INT_MIN;
	int min_y = INT_MAX, max_y = INT_MIN;
	for (const Cell& cell : pattern.cells) {
		cells.insert(cell);
		if (cell.x < min_x) min_x = cell.x;
		if (cell.x > max_x) max_x = cell.x;
		if (cell.y < min_y) min_y = cell.y;
		if (cell.y > max_y) max_y = cell.y;
	}
	// Centre the pattern
	int pattern_width_px = (max_x - min_x + 1) * CELL_SIZES[cell_size_idx];
	int pattern_height_px = (max_y - min_y + 1) * CELL_SIZES[cell_size_idx];
	view_offset.x = (GRID_WIDTH - pattern_width_px) / 2;
	view_offset.y = (GRID_HEIGHT - pattern_height_px) / 2;

	generation_num = 0;
	num_removed_cells = 0;
}


void next_generation() {
	live_neighbour_counts.clear();
	new_cells.clear();

	for (const Cell& cell : cells) {
		for (int dx = -1; dx <= 1; dx++) {
			for (int dy = -1; dy <= 1; dy++) {
				if (dx == 0 && dy == 0)
					continue;  // Skip the cell itself
				neighbour_cell = {cell.x + dx, cell.y + dy};
				live_neighbour_counts[neighbour_cell]++;
			}
		}
	}

	for (const auto& [pos, count] : live_neighbour_counts)
		if (count == 3 || (count == 2 && cells.count(pos)))
			new_cells.insert(pos);

	// Every 1000 generations, remove cells that are very far away
	if (generation_num % 1000 == 0) {
		for (auto it = new_cells.begin(); it != new_cells.end();) {
			if (abs(it->x) > 10000 || abs(it->y) > 10000) {
				it = new_cells.erase(it);
				num_removed_cells++;
			} else {
				it++;
			}
		}
	}

	cells.swap(new_cells);
	generation_num++;
}


void draw() {
	window.clear(sf::Color(40, 40, 40));

	for (const Cell& cell : cells) {
		sf::RectangleShape rect(sf::Vector2f(CELL_SIZES[cell_size_idx], CELL_SIZES[cell_size_idx]));
		rect.setPosition(
			cell.x * CELL_SIZES[cell_size_idx] + view_offset.x,
			cell.y * CELL_SIZES[cell_size_idx] + view_offset.y + LABEL_HEIGHT
		);
		rect.setFillColor(sf::Color(220, 140, 0));
		window.draw(rect);
	}

	sf::RectangleShape lbl_area(sf::Vector2f(GRID_WIDTH, LABEL_HEIGHT));
	lbl_area.setPosition(0, 0);
	lbl_area.setFillColor(sf::Color::Black);
	window.draw(lbl_area);

	sf::Text header_text("P: select a preset pattern  |  Z/X: cycle through patterns  |  R: randomise cells  |  S: show info  |  Up/down: change delay  |  Space: play/pause  |  Mouse drag: move around  |  Scroll: zoom", font, 13);
	sf::FloatRect header_text_rect = header_text.getLocalBounds();
	header_text.setOrigin(int(header_text_rect.left + header_text_rect.width / 2), int(header_text_rect.top + header_text_rect.height / 2));
	header_text.setPosition(GRID_WIDTH / 2, LABEL_HEIGHT / 2);
	header_text.setFillColor(sf::Color::White);
	window.draw(header_text);

	if (show_info) {
		sf::Text data_text(
			"Generation: " + std::to_string(generation_num)
			+ "\nPopulation: " + std::to_string(cells.size() + num_removed_cells)
			+ "\nPattern: " + (pattern_idx == -1 ? "Random" : ALL_PATTERNS[pattern_idx].description)
			+ "\nDelay: " + std::to_string(DELAYS[delay_idx]) + "ms",
			font,
			14
		);
		data_text.setPosition(14, 40);
		data_text.setFillColor(sf::Color::White);
		window.draw(data_text);
	}

	window.display();
}


int main() {
	bool paused = false;
	bool left_btn_down = false;
	int screenshot_counter = 0;
	sf::Vector2f drag_origin, mouse_pos, grid_mouse_pos;
	sf::Clock simulation_clock;
	sf::Event event;

	font.loadFromFile("C:/Windows/Fonts/consola.ttf");
	randomise_live_cells();

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;

				case sf::Event::KeyPressed:
					switch (event.key.code) {
						case sf::Keyboard::P:
							std::cout << "Select a pattern:\n";
							for (int i = 0; i < ALL_PATTERNS.size(); i++)
								std::cout << (i + 1) << ": " << ALL_PATTERNS[i].description << '\n';
							std::cout << ">>> ";
							std::cin >> pattern_idx;
							pattern_idx--;
							set_pattern(ALL_PATTERNS[pattern_idx]);
							paused = true;
							break;
						case sf::Keyboard::Z:
							pattern_idx = (--pattern_idx + ALL_PATTERNS.size()) % ALL_PATTERNS.size();
							set_pattern(ALL_PATTERNS[pattern_idx]);
							paused = true;
							break;
						case sf::Keyboard::X:
							pattern_idx = (++pattern_idx + ALL_PATTERNS.size()) % ALL_PATTERNS.size();
							set_pattern(ALL_PATTERNS[pattern_idx]);
							paused = true;
							break;
						case sf::Keyboard::R:
							randomise_live_cells();
							view_offset = {0, 0};
							break;
						case sf::Keyboard::S:
							show_info = !show_info;
							break;
						case sf::Keyboard::Up:
							delay_idx++;
							if (delay_idx >= sizeof(DELAYS) / sizeof(int))
								delay_idx = 0;
							break;
						case sf::Keyboard::Down:
							delay_idx--;
							if (delay_idx < 0)
								delay_idx = sizeof(DELAYS) / sizeof(int) - 1;
							break;
						case sf::Keyboard::Space:
							paused = !paused;
							break;
					}
					break;

				case sf::Event::MouseButtonPressed:
					if (event.mouseButton.button == sf::Mouse::Left) {
						left_btn_down = true;
						drag_origin = sf::Vector2f(sf::Mouse::getPosition(window)) - view_offset;
					}
					break;

				case sf::Event::MouseMoved:
					if (left_btn_down)
						view_offset = sf::Vector2f(sf::Mouse::getPosition(window)) - drag_origin;
					break;

				case sf::Event::MouseButtonReleased:
					if (event.mouseButton.button == sf::Mouse::Left)
						left_btn_down = false;
					break;

				case sf::Event::MouseWheelScrolled:
					if (event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel) {
						// Zoom in/out about the mouse position

						mouse_pos = sf::Vector2f(sf::Mouse::getPosition(window));
						grid_mouse_pos = {
							(mouse_pos.x - view_offset.x) / float(CELL_SIZES[cell_size_idx]),
							(mouse_pos.y - view_offset.y - LABEL_HEIGHT) / float(CELL_SIZES[cell_size_idx])
						};

						if (event.mouseWheelScroll.delta > 0) {
							// Zoom in
							if (cell_size_idx < sizeof(CELL_SIZES) / sizeof(int) - 1)
								cell_size_idx++;
							else
								continue;
						} else {
							// Zoom out
							if (cell_size_idx > 0)
								cell_size_idx--;
							else
								continue;
						}

						view_offset.x = mouse_pos.x - grid_mouse_pos.x * CELL_SIZES[cell_size_idx];
						view_offset.y = mouse_pos.y - grid_mouse_pos.y * CELL_SIZES[cell_size_idx] - LABEL_HEIGHT;
					}
					break;
			}
		}

		draw();
		if (!paused && simulation_clock.getElapsedTime().asMilliseconds() >= DELAYS[delay_idx]) {
			next_generation();
			simulation_clock.restart();
		}

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
