/*
Cell class used by both adjacent tiling and overlapping tiling

Author: Sam Barba
Created 19/07/2023
*/

#ifndef CELL
#define CELL

#include <numeric>
#include <random>
#include <vector>


std::random_device rd;
std::mt19937 gen(rd());


class Cell {
	public:
		int y;
		int x;
		std::vector<int> tile_options;

		Cell(const int y = 0, const int x = 0, const int num_tile_options = 1) {
			this->y = y;
			this->x = x;
			tile_options.resize(num_tile_options);
			std::iota(tile_options.begin(), tile_options.end(), 0);
		}

		int entropy() {
			return tile_options.size();
		}

		bool is_collapsed() {
			return entropy() == 1;
		}

		bool observe() {
			// By observing a cell, we collapse it into 1 possible state. A 'contradiction' boolean is returned
			// (true if surrounding tiles mean that there are no options left, false otherwise).

			if (entropy() > 0) {
				std::uniform_int_distribution<int> dist(0, entropy() - 1);
				int rand_choice = tile_options[dist(gen)];
				tile_options.clear();
				tile_options.push_back(rand_choice);
				return false;
			}

			return true;
		}
};

#endif
