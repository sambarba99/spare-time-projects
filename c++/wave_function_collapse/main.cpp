/*
Visualisation of Wave Function Collapse (via adjacent tiling or overlapping tiling algorithm)

Author: Sam Barba
Created 19/07/2023
*/

#include "adjacent/adjacent_main.h"
#include "overlapping/overlapping_main.h"


const string TILING_MODE = "adjacent";  // "adjacent" or "overlapping"

// If running adjacent tiling, choose a dir from /adjacent/tile_imgs (circuit, pipes, water).
// For overlapping mode, choose a source image from /overlapping/src_imgs (flowers, island, link, skyline, spirals, waves).
const string IMG_TYPE = "circuit";


int main() {
	if (TILING_MODE == "adjacent")
		adjacent::main(IMG_TYPE);
	else if (TILING_MODE == "overlapping")
		overlapping::main(IMG_TYPE);
	else
		std::cout << "TILING_MODE must be \"adjacent\" or \"overlapping\" (got: " << TILING_MODE << ")\n";

	return 0;
}
