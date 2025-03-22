/*
Cell class

Author: Sam Barba
Created 19/07/2023
*/


#include <numeric>
#include <random>
#include <vector>


std::random_device rd;
std::mt19937 gen(rd());


class Cell {
	public:
		int y;
		int x;
		std::vector<int> tileOptions;

		Cell(const int y = 0, const int x = 0, const int numTileOptions = 1) {
			this->y = y;
			this->x = x;
			tileOptions.resize(numTileOptions);
			std::iota(tileOptions.begin(), tileOptions.end(), 0);
		}

		int entropy() {
			return tileOptions.size();
		}

		bool isCollapsed() {
			return entropy() == 1;
		}

		bool observe() {
			// By observing a cell, we collapse it into 1 possible state. A 'contradiction' boolean is returned
			// (true if surrounding tiles mean that there are no options left, false otherwise).

			if (entropy() > 0) {
				std::uniform_int_distribution<int> dist(0, entropy() - 1);
				int randOption = tileOptions[dist(gen)];
				tileOptions.clear();
				tileOptions.push_back(randOption);
				return false;
			}

			return true;
		}
};
