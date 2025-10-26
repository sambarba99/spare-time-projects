/*
Visualisation of Wave Function Collapse (overlapping tiling algorithm)

1. First, an input image from ./src_imgs is read (change via COLLAGE_TYPE in ../main.cpp).
2. Patches (tiles) of a predefined size are extracted from the image into a list. E.g. with a source image of size
20x20, this tile list would be 400 long.
3. Tile objects are created using these images, and adjacency rules are created by using the colours on the edges of
each tile (TILE_SIZE - 1 wide) and comparing the overlap with the edges of other tiles.
4. A grid of cells (../cell.h) is then initiated, each one initially having multiple possible states (tiles). This is
their superposition. By iterating WFC and checking which neighbour tile states are allowed, each cell will eventually
have just one state (its superposition is 'collapsed'), meaning its tile image can be visualised.

Author: Sam Barba
Created 15/02/2025
*/

#ifndef OVERLAPPING_MAIN
#define OVERLAPPING_MAIN

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <SFML/Graphics.hpp>

#include "../cell.h"

using std::string;
using std::vector;


namespace overlapping {
	const int COLLAGE_WIDTH = 41;  // In tiles
	const int COLLAGE_HEIGHT = 25;
	const int SAMPLE_TILE_SIZE = 3;
	const int CELL_SIZE = 20;
	const int MAX_DEPTH = 20;
	const bool RENDER_ENTROPY_VALUES = true;
	const int FPS = 60;
};

class OverlappingModeTile {
	public:
		sf::Texture texture;
		string north_edge, east_edge, south_edge, west_edge;
		std::map<string, vector<int>> neighbour_options;

		OverlappingModeTile(const sf::Texture& texture, const vector<string>& edge_colour_codes) {
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

		void generate_adjacency_rules(const vector<OverlappingModeTile>& tiles) {
			for (int idx = 0; idx < tiles.size(); idx++) {
				OverlappingModeTile other = tiles[idx];
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


int overlapping_tiling(const std::string collage_type) {
	generate_tiles(collage_type);
	tile_size = tiles[0].texture.getSize().x;

	sf::RenderWindow window(
		sf::VideoMode(overlapping::COLLAGE_WIDTH * tile_size, overlapping::COLLAGE_HEIGHT * tile_size),
		"Wave Function Collapse (overlapping tiling algorithm)"
	);
	window.setFramerateLimit(overlapping::FPS);
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");

	vector<vector<Cell>> grid(overlapping::COLLAGE_HEIGHT, vector<Cell>(overlapping::COLLAGE_WIDTH));
	bool starting = true;

	while (window.isOpen()) {
		while (window.pollEvent(event))
			if (event.type == sf::Event::Closed)
				window.close();

		if (starting) {
			for (int y = 0; y < overlapping::COLLAGE_HEIGHT; y++)
				for (int x = 0; x < overlapping::COLLAGE_WIDTH; x++)
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
