/*
Visualisation of Wave Function Collapse (adjacent tiling algorithm)

1. First, a set of tile images is loaded from ./tile_imgs (change via COLLAGE_TYPE in ../main.cpp).
2. Tile objects are created using these images, together with predefined edge colour codes (north/east/south/west edges,
read clockwise) and the number of possible orientations of the tile.
3. Adjacency rules are created by using the edge colour codes of the tiles.
4. A grid of cells (../cell.h) is then initiated, each one initially having multiple possible states (tiles). This is
their superposition. By iterating WFC and checking which neighbour tile states are allowed, each cell will eventually
have just one state (its superposition is 'collapsed'), meaning its tile image can be visualised.

Author: Sam Barba
Created 19/07/2023
*/

#ifndef ADJACENT_MAIN
#define ADJACENT_MAIN

#include <filesystem>
#include <iostream>
#include <random>
#include <SFML/Graphics.hpp>

#include "../cell.h"

using std::string;
using std::vector;


namespace adjacent {
	const int COLLAGE_WIDTH = 49;  // In tiles
	const int COLLAGE_HEIGHT = 29;
	const bool RENDER_ENTROPY_VALUES = true;
	const int FPS = 60;
	const int DIRECTIONS[4][2] = {
		{-1, 0},  // N
		{0, 1},  // E
		{1, 0},  // S
		{0, -1}  // W
	};
	const string OPPOSITES[4] = {"south", "west", "north", "east"};
};

class AdjacentModeTile {
	public:
		sf::Texture texture;
		string north_edge, east_edge, south_edge, west_edge;
		std::map<string, vector<int>> neighbour_options;

		AdjacentModeTile(const sf::Texture& texture, const vector<string>& edge_colour_codes) {
			this->texture = texture;
			north_edge = edge_colour_codes[0];
			east_edge = edge_colour_codes[1];
			south_edge = edge_colour_codes[2];
			west_edge = edge_colour_codes[3];
			neighbour_options["north"] = {};
			neighbour_options["east"] = {};
			neighbour_options["south"] = {};
			neighbour_options["west"] = {};
		}

		void generate_adjacency_rules(const vector<AdjacentModeTile>& tiles) {
			for (int idx = 0; idx < tiles.size(); idx++) {
				AdjacentModeTile other = tiles[idx];
				if (std::equal(north_edge.begin(), north_edge.end(), other.south_edge.rbegin()))
					neighbour_options["north"].push_back(idx);
				if (std::equal(east_edge.begin(), east_edge.end(), other.west_edge.rbegin()))
					neighbour_options["east"].push_back(idx);
				if (std::equal(south_edge.begin(), south_edge.end(), other.north_edge.rbegin()))
					neighbour_options["south"].push_back(idx);
				if (std::equal(west_edge.begin(), west_edge.end(), other.east_edge.rbegin()))
					neighbour_options["west"].push_back(idx);
			}
		}
};

int tile_size;
vector<AdjacentModeTile> tiles;
sf::Event event;
sf::Font font;


void generate_tiles(const string collage_type) {
	// Create Tile objects by reading image files

	tiles.clear();
	string dir = "./adjacent/tile_imgs/" + collage_type;

	for (const auto& entry : std::filesystem::directory_iterator(dir)) {
		if (entry.path().extension() == ".png") {
			string filename = entry.path().filename().string();
			std::istringstream ss(filename);
			vector<string> parts;
			string part;

			while (std::getline(ss, part, '_'))
				parts.push_back(part);

			vector<string> edge_colour_codes(parts.end() - 5, parts.end() - 1);
			int num_orientations = std::stoi(parts.back().substr(0, parts.back().find('.')));

			sf::Texture texture;
			texture.loadFromFile(entry.path().string());
			tiles.push_back(AdjacentModeTile(texture, edge_colour_codes));

			if (num_orientations > 1) {
				vector<string> old_edges = edge_colour_codes;
				for (int n = 1; n < num_orientations; n++) {
					sf::RenderTexture render_texture;
					render_texture.create(texture.getSize().x, texture.getSize().y);
					sf::Sprite sprite(texture);
					sprite.setOrigin(texture.getSize().x / 2, texture.getSize().y / 2);
					sprite.setPosition(texture.getSize().x / 2, texture.getSize().y / 2);  // Center it
					sprite.setRotation(n * 90);  // Rotate by n * 90 degrees
					render_texture.clear(sf::Color::Transparent);
					render_texture.draw(sprite);
					render_texture.display();

					sf::Texture rotated_texture = render_texture.getTexture();
					vector<string> new_edges(4);
					for (int i = 0; i < 4; i++)
						new_edges[i] = old_edges[(i - n + 4) % 4];

					tiles.push_back(AdjacentModeTile(rotated_texture, new_edges));
				}
			}
		}
	}

	for (AdjacentModeTile& tile : tiles)
		tile.generate_adjacency_rules(tiles);
}


void draw(sf::RenderWindow& window, vector<vector<Cell>>& grid) {
	window.clear();

	for (vector<Cell>& row : grid)
		for (Cell& cell : row) {
			if (cell.is_collapsed()) {
				int tile_idx = cell.tile_options[0];
				sf::Sprite sprite;
				sprite.setTexture(tiles[tile_idx].texture);
				sprite.setScale(-1, -1);
				sprite.setPosition(cell.x * tile_size + tile_size, cell.y * tile_size + tile_size);
				window.draw(sprite);
			} else if (adjacent::RENDER_ENTROPY_VALUES) {
				sf::Text cell_text(std::to_string(cell.entropy()), font, 11);
				cell_text.setPosition(int(float(cell.x + 0.21f) * tile_size), int(float(cell.y + 0.19f) * tile_size));
				cell_text.setFillColor(sf::Color::White);
				window.draw(cell_text);
			}
		}

	window.display();
}


string wave_function_collapse(sf::RenderWindow& window, vector<vector<Cell>>& grid, const bool is_first_cell = false) {
	// 1. Choose a cell whose superposition to collapse into one state

	Cell* next_cell_to_collapse = nullptr;

	if (is_first_cell) {
		next_cell_to_collapse = &grid[adjacent::COLLAGE_HEIGHT / 2][adjacent::COLLAGE_WIDTH / 2];  // Start in centre
	} else {
		// From all the non-collapsed cells, choose the one with minimum entropy, breaking any ties randomly
		vector<Cell*> uncollapsed;
		for (vector<Cell>& row : grid)
			for (Cell& cell : row)
				if (!cell.is_collapsed())
					uncollapsed.push_back(&cell);

		if (!uncollapsed.empty()) {
			std::sort(
				uncollapsed.begin(),
				uncollapsed.end(),
				[](Cell* a, Cell* b) { return a->entropy() < b->entropy(); }
			);
		} else {
			// If all are collapsed (1 state option left), we're done
			std::cout << "All superpositions collapsed to 1 state, starting again\n";
			draw(window, grid);
			return "all cells collapsed";
		}

		// Collect cells with minimum entropy
		int min_entropy = uncollapsed.front()->entropy();
		vector<Cell*> min_entropy_cells;
		for (Cell* cell : uncollapsed)
			if (cell->entropy() == min_entropy)
				min_entropy_cells.push_back(cell);

		// Randomly choose one
		std::uniform_int_distribution<int> dist(0, static_cast<int>(min_entropy_cells.size()) - 1);
		next_cell_to_collapse = min_entropy_cells[dist(gen)];
	}

	bool contradiction = next_cell_to_collapse->observe();
	if (contradiction) {
		std::cout << "Reached contradiction, starting again\n";
		return "contradiction";
	}

	// 2. Propagate adjacency rules to ensure only legal superpositions remain

	vector<Cell*> stack;
	stack.push_back(next_cell_to_collapse);

	while (!stack.empty()) {
		Cell* current = stack.back();
		stack.pop_back();

		for (int d = 0; d < 4; d++) {
			int dy = adjacent::DIRECTIONS[d][0];
			int dx = adjacent::DIRECTIONS[d][1];
			// Adjacent cell coords
			int adj_y = current->y + dy;
			int adj_x = current->x + dx;

			if (adj_y < 0 || adj_y >= adjacent::COLLAGE_HEIGHT || adj_x < 0 || adj_x >= adjacent::COLLAGE_WIDTH)
				continue;

			Cell* adj = &grid[adj_y][adj_x];
			const string opposite = adjacent::OPPOSITES[d];

			// Iterate over all tile options of adjacent cell, removing invalid ones
			bool changed = false;
			vector<int> new_options;
			for (int adj_idx : adj->tile_options) {
				const AdjacentModeTile& adj_tile = tiles[adj_idx];
				bool allowed = false;

				// Check if at least one tile in current cell is compatible
				for (int cur_idx : current->tile_options) {
					const AdjacentModeTile& cur_tile = tiles[cur_idx];
					const auto& allowed_list = cur_tile.neighbour_options.at(opposite);
					if (std::find(allowed_list.begin(), allowed_list.end(), adj_idx) != allowed_list.end()) {
						allowed = true;
						break;
					}
				}

				if (allowed)
					new_options.push_back(adj_idx);
			}

			if (new_options.size() != adj->tile_options.size()) {
				changed = true;
				adj->tile_options = std::move(new_options);
				if (adj->entropy() == 0) {
					std::cout << "Reached contradiction, starting again\n";
					return "contradiction";
				}
				stack.push_back(adj);  // Push updated adjacent cell to stack
			}
		}
	}

	draw(window, grid);
	if (is_first_cell)
		sf::sleep(sf::milliseconds(1000));

	return "propagated rules";
}


int adjacent_tiling(const string collage_type) {
	generate_tiles(collage_type);
	tile_size = tiles[0].texture.getSize().x;

	sf::RenderWindow window(
		sf::VideoMode(adjacent::COLLAGE_WIDTH * tile_size, adjacent::COLLAGE_HEIGHT * tile_size),
		"Wave Function Collapse (adjacent tiling algorithm)"
	);
	window.setFramerateLimit(adjacent::FPS);
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");

	vector<vector<Cell>> grid(adjacent::COLLAGE_HEIGHT, vector<Cell>(adjacent::COLLAGE_WIDTH));
	bool starting = true;

	while (window.isOpen()) {
		while (window.pollEvent(event))
			if (event.type == sf::Event::Closed)
				window.close();

		if (starting) {
			for (int y = 0; y < adjacent::COLLAGE_HEIGHT; y++)
				for (int x = 0; x < adjacent::COLLAGE_WIDTH; x++)
					grid[y][x] = Cell(y, x, tiles.size());

			draw(window, grid);
			sf::sleep(sf::milliseconds(1000));
			wave_function_collapse(window, grid, true);
			starting = false;
		} else {
			string result = wave_function_collapse(window, grid);
			if (result == "contradiction" || result == "all cells collapsed") {
				// sf::Texture texture;
				// sf::Image screenshot;
				// texture.create(window.getSize().x, window.getSize().y);
				// texture.update(window);
				// screenshot = texture.copyToImage();
				// string file_path = "./" + collage_type + ".png";
				// screenshot.saveToFile(file_path);

				sf::sleep(sf::milliseconds(2000));
				starting = true;
			}
		}
	}

	return 0;
}

#endif
