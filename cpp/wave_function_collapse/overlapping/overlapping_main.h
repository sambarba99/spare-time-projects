/*
Visualisation of Wave Function Collapse (overlapping tiling algorithm)

1. First, an input image from ./src_imgs is read (change via IMG_TYPE in ../main.cpp).
2. Patches (tiles) of a predefined size are extracted from the image into a list. E.g. with a source image of size
20x20, this tile list would be 400 long.
3. Tile objects are created using these images, and adjacency rules are created by using the colours on the edges of
each tile (SAMPLE_TILE_SIZE - 1 wide) and comparing the overlap with the edges of other tiles.
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

using cv::Mat;
using cv::Vec3b;
using std::string;
using std::vector;


namespace overlapping {
	const int COLLAGE_WIDTH = 41;  // In tiles
	const int COLLAGE_HEIGHT = 25;
	const int SAMPLE_TILE_SIZE = 3;
	const int CELL_SIZE = 20;
	const int MAX_DEPTH = 20;
	const bool RENDER_ENTROPY_VALUES = false;
	const int FPS = 60;

	bool pixels_equal(Mat in1, Mat in2) {
		Mat diff;
		cv::absdiff(in1, in2, diff);
		return cv::countNonZero(diff.reshape(1)) == 0;
	}

	class Tile {
		public:
			Mat img;
			Vec3b centre_pixel;
			std::unordered_map<Dir, vector<int>> neighbour_options;

			Tile(const Mat& img) {
				this->img = img;
				int centre_y = img.rows / 2;
				int centre_x = img.cols / 2;
				centre_pixel = img.at<Vec3b>(centre_y, centre_x);
			}

			void generate_adjacency_rules(const vector<Tile*>& tiles) {
				Mat north_colours = img.rowRange(0, img.rows - 1);
				Mat east_colours = img.colRange(1, img.cols);
				Mat south_colours = img.rowRange(1, img.rows);
				Mat west_colours = img.colRange(0, img.cols - 1);

				for (int i = 0; i < tiles.size(); i++) {
					const Tile* other = tiles[i];

					Mat other_north_colours = other->img.rowRange(0, other->img.rows - 1);
					Mat other_east_colours = other->img.colRange(1, other->img.cols);
					Mat other_south_colours = other->img.rowRange(1, other->img.rows);
					Mat other_west_colours = other->img.colRange(0, other->img.cols - 1);

					if (pixels_equal(north_colours, other_south_colours))
						neighbour_options[Dir::North].emplace_back(i);
					if (pixels_equal(east_colours, other_west_colours))
						neighbour_options[Dir::East].emplace_back(i);
					if (pixels_equal(south_colours, other_north_colours))
						neighbour_options[Dir::South].emplace_back(i);
					if (pixels_equal(west_colours, other_east_colours))
						neighbour_options[Dir::West].emplace_back(i);
				}
			}
	};

	vector<Tile*> tiles;
	sf::Event event;
	sf::Font font;


	void generate_tiles(const string img_type) {
		Mat src_img = cv::imread("./overlapping/src_imgs/" + img_type + ".png");

		cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
		int h = src_img.rows;
		int w = src_img.cols;

		// Create Tile objects by looping through all possible SAMPLE_TILE_SIZE x SAMPLE_TILE_SIZE
		// tiles in the source image, wrapping around edges

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				Mat tile(SAMPLE_TILE_SIZE, SAMPLE_TILE_SIZE, src_img.type());
				for (int ty = 0; ty < SAMPLE_TILE_SIZE; ty++) {
					int y_idx = (y + ty) % h;
					for (int tx = 0; tx < SAMPLE_TILE_SIZE; tx++) {
						int x_idx = (x + tx) % w;
						tile.at<Vec3b>(ty, tx) = src_img.at<Vec3b>(y_idx, x_idx);
					}
				}
				tiles.emplace_back(new Tile(tile));
			}
		}

		// Show the source image and a collage of all the extracted tiles

		// int new_h = h * 16;
		// int new_w = w * 16;
		// Mat src_img_zoomed;
		// cv::resize(src_img, src_img_zoomed, cv::Size(new_w, new_h), 0, 0, cv::INTER_NEAREST);
		// cv::cvtColor(src_img_zoomed, src_img_zoomed, cv::COLOR_RGB2BGR);
		// cv::imshow(img_type + ".png (" + std::to_string(w) + "x" + std::to_string(h) + ")", src_img_zoomed);
		// cv::imwrite("./source.png", src_img_zoomed);

		// int collage_h = h * SAMPLE_TILE_SIZE + h - 1;
		// int collage_w = w * SAMPLE_TILE_SIZE + w - 1;
		// Mat extracted_tile_collage(collage_h, collage_w, CV_8UC3, cv::Scalar(13, 17, 23));

		// for (int i = 0; i < tiles.size(); i++) {
		// 	int row = i / w;
		// 	int col = i % w;
		// 	int y = row * (SAMPLE_TILE_SIZE + 1);  // Space tiles by 1px
		// 	int x = col * (SAMPLE_TILE_SIZE + 1);
		// 	cv::Rect roi(x, y, SAMPLE_TILE_SIZE, SAMPLE_TILE_SIZE);  // Region of interest
		// 	tiles[i]->img.copyTo(extracted_tile_collage(roi));
		// }

		// Mat extracted_tile_collage_zoomed;
		// cv::resize(extracted_tile_collage, extracted_tile_collage_zoomed, cv::Size(collage_w * 8, collage_h * 8), 0, 0, cv::INTER_NEAREST);
		// cv::cvtColor(extracted_tile_collage_zoomed, extracted_tile_collage_zoomed, cv::COLOR_RGB2BGR);
		// cv::imshow("All " + std::to_string(SAMPLE_TILE_SIZE) + "x" + std::to_string(SAMPLE_TILE_SIZE) + " tiles", extracted_tile_collage_zoomed);
		// cv::imwrite("./all_tiles.png", extracted_tile_collage_zoomed);
		// cv::waitKey();
		// cv::destroyAllWindows();

		for (Tile* tile : tiles)
			tile->generate_adjacency_rules(tiles);
	}


	void draw(sf::RenderWindow& window, const vector<vector<Cell*>>& grid) {
		window.clear();

		for (const vector<Cell*>& row : grid) {
			for (Cell* cell : row) {
				Vec3b tile_colour;
				if (cell->is_collapsed()) {
					int tile_idx = cell->tile_options[0];
					tile_colour = tiles[tile_idx]->centre_pixel;
				} else {
					cv::Vec3f mean_colour(0.f, 0.f, 0.f);
					for (int tile_idx : cell->tile_options)
						mean_colour += cv::Vec3f(tiles[tile_idx]->centre_pixel);
					mean_colour /= static_cast<float>(cell->tile_options.size());
					tile_colour = Vec3b(
						static_cast<uchar>(std::clamp(mean_colour[0], 0.f, 255.f)),
						static_cast<uchar>(std::clamp(mean_colour[1], 0.f, 255.f)),
						static_cast<uchar>(std::clamp(mean_colour[2], 0.f, 255.f))
					);
				}
				sf::RectangleShape rect(sf::Vector2f(CELL_SIZE, CELL_SIZE));
				rect.setPosition(cell->x * CELL_SIZE, cell->y * CELL_SIZE);
				rect.setFillColor(sf::Color(tile_colour[0], tile_colour[1], tile_colour[2]));
				window.draw(rect);

				if (!cell->is_collapsed() && RENDER_ENTROPY_VALUES) {
					sf::Text cell_text(std::to_string(cell->entropy()), font, 10);
					cell_text.setPosition(int(float(cell->x + 0.1f) * CELL_SIZE), int(float(cell->y + 0.19f) * CELL_SIZE));
					cell_text.setFillColor(sf::Color(
						255 - tile_colour[0],
						255 - tile_colour[1],
						255 - tile_colour[2]
					));
					window.draw(cell_text);
				}
			}
		}

		window.display();
	}


	string wave_function_collapse(const string img_type, sf::RenderWindow& window, const vector<vector<Cell*>>& grid, const bool is_first_cell = false) {
		// 1. Choose a cell whose superposition to collapse into one state

		Cell* next_cell_to_collapse = nullptr;

		if (is_first_cell) {
			// If 'flowers' or 'skyline', start on the ground (bottom of grid). Otherwise, start in the grid centre.
			if (img_type == "flowers") {
				next_cell_to_collapse = grid[COLLAGE_HEIGHT - 1][COLLAGE_WIDTH / 2];
				next_cell_to_collapse->tile_options = {next_cell_to_collapse->tile_options[336]};
			} else if (img_type == "skyline") {
				next_cell_to_collapse = grid[COLLAGE_HEIGHT - 1][COLLAGE_WIDTH / 2];
				next_cell_to_collapse->tile_options = {next_cell_to_collapse->tile_options[401]};
			} else {
				next_cell_to_collapse = grid[COLLAGE_HEIGHT / 2][COLLAGE_WIDTH / 2];
			}
		} else {
			// From all the non-collapsed cells, choose the one with minimum entropy, breaking any ties randomly
			vector<Cell*> uncollapsed;
			for (const vector<Cell*>& row : grid)
				for (Cell* cell : row)
					if (!cell->is_collapsed())
						uncollapsed.emplace_back(cell);

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
					min_entropy_cells.emplace_back(cell);

			if (min_entropy_cells.size() == 1) {
				next_cell_to_collapse = min_entropy_cells[0];
			} else {
				// Randomly choose one
				std::uniform_int_distribution<int> dist(0, static_cast<int>(min_entropy_cells.size()) - 1);
				next_cell_to_collapse = min_entropy_cells[dist(gen)];
			}
		}

		bool contradiction = next_cell_to_collapse->observe();
		if (contradiction) {
			std::cout << "Reached contradiction, starting again\n";
			return "contradiction";
		}

		// 2. Propagate adjacency rules to ensure only legal superpositions remain

		vector<std::pair<Cell*, int>> stack;
		stack.emplace_back(next_cell_to_collapse, 0);

		while (!stack.empty()) {
			Cell* current = stack.back().first;
			int depth = stack.back().second;
			stack.pop_back();

			if (depth > MAX_DEPTH)
				continue;

			for (int d = 0; d < 4; d++) {
				// Adjacent cell coords
				auto [dy, dx] = DIRECTION_VECTORS[d];
				int adj_y = current->y + dy;
				int adj_x = current->x + dx;

				if (adj_y < 0 || adj_y >= COLLAGE_HEIGHT || adj_x < 0 || adj_x >= COLLAGE_WIDTH)
					continue;

				Cell* adj = grid[adj_y][adj_x];
				const Dir opposite = OPPOSITE_DIRS[d];

				// Iterate over all tile options of adjacent cell, removing invalid ones
				vector<int> new_options;
				for (int adj_idx : adj->tile_options) {
					const Tile* adj_tile = tiles[adj_idx];
					bool allowed = false;

					// Check if at least one tile in current cell is compatible
					for (int cur_idx : current->tile_options) {
						const Tile* cur_tile = tiles[cur_idx];
						const auto& allowed_list = cur_tile->neighbour_options.at(opposite);
						if (std::find(allowed_list.begin(), allowed_list.end(), adj_idx) != allowed_list.end()) {
							allowed = true;
							break;
						}
					}

					if (allowed)
						new_options.emplace_back(adj_idx);
				}

				if (new_options.size() != adj->tile_options.size()) {
					adj->tile_options = std::move(new_options);
					if (adj->entropy() == 0) {
						std::cout << "Reached contradiction, starting again\n";
						return "contradiction";
					}
					stack.emplace_back(adj, depth + 1);  // Add updated adjacent cell to stack, together with next level of depth
				}
			}
		}

		draw(window, grid);
		if (is_first_cell)
			sf::sleep(sf::milliseconds(1000));

		return "propagated rules";
	}


	int main(const string img_type) {
		generate_tiles(img_type);

		sf::RenderWindow window(
			sf::VideoMode(COLLAGE_WIDTH * CELL_SIZE, COLLAGE_HEIGHT * CELL_SIZE),
			"Wave Function Collapse (overlapping tiling algorithm)"
		);
		window.setFramerateLimit(FPS);
		font.loadFromFile("C:/Windows/Fonts/consola.ttf");

		vector<vector<Cell*>> grid(COLLAGE_HEIGHT, vector<Cell*>(COLLAGE_WIDTH));
		bool starting = true;

		while (window.isOpen()) {
			while (window.pollEvent(event))
				if (event.type == sf::Event::Closed)
					window.close();

			if (starting) {
				for (int y = 0; y < COLLAGE_HEIGHT; y++)
					for (int x = 0; x < COLLAGE_WIDTH; x++)
						grid[y][x] = new Cell(y, x, tiles.size());

				draw(window, grid);
				sf::sleep(sf::milliseconds(1000));
				wave_function_collapse(img_type, window, grid, true);
				starting = false;
			} else {
				string result = wave_function_collapse(img_type, window, grid);
				if (result == "contradiction" || result == "all cells collapsed") {
					// sf::Texture texture;
					// sf::Image screenshot;
					// texture.create(window.getSize().x, window.getSize().y);
					// texture.update(window);
					// screenshot = texture.copyToImage();
					// string file_path = "./" + img_type + ".png";
					// screenshot.saveToFile(file_path);

					sf::sleep(sf::milliseconds(2000));
					starting = true;
				}
			}
		}

		return 0;
	}
};

#endif
