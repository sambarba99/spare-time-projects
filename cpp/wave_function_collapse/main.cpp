/*
Visualisation of Wave Function Collapse (via adjacent tiling or overlapping tiling algorithm)

Author: Sam Barba
Created 19/07/2023
*/

#include "adjacent/adjacent_main.h"
#include "overlapping/overlapping_main.h"


const bool ADJACENT_MODE = true;  // Set to true to demo the adjacent algorithm, false for overlapping algorithm

// If ADJACENT_MODE, choose a dir from /adjacent/tile_imgs.
// Otherwise, choose a source image from /overlapping/src_imgs.
const string COLLAGE_TYPE = "water";


int main() {
	if (ADJACENT_MODE)
		adjacent_tiling(COLLAGE_TYPE);
	else
		overlapping_tiling(COLLAGE_TYPE);
}
