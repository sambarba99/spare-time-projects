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


enum class Dir : int {
	North = 0,
	East = 1,
	South = 2,
	West = 3
};
const int DIRECTION_VECTORS[4][2] = {
	{-1, 0},  // N
	{0, 1},  // E
	{1, 0},  // S
	{0, -1}  // W
};
const Dir OPPOSITE_DIRS[4] = {
	Dir::South,  // N -> S
	Dir::West,  // E -> W
	Dir::North,  // S -> N
	Dir::East  // W -> E
};

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
				tile_options.emplace_back(rand_choice);
				return false;
			}

			return true;
		}
};

#endif
