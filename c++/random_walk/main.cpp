/*
Random walk generator

Author: Sam Barba
Created 18/10/2024
*/

#include <opencv2/opencv.hpp>
#include <random>


const int NUM_STEPS = 1e6;
const int BORDER = 10;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dist(0, 3);


int main() {
	// 1. Generate a walk

	std::cout << "Walking...\n";

	std::vector<std::pair<int, int>> walk_coords(NUM_STEPS + 1);
	int y = 0, x = 0;
	int dir;
	int min_x = INT_MAX, max_x = INT_MIN;
	int min_y = INT_MAX, max_y = INT_MIN;

	for (int i = 0; i <= NUM_STEPS; i++) {
		walk_coords[i] = {y, x};

		dir = dist(gen);

		if (dir == 0) y--;  // North
		else if (dir == 1) x++;  // East
		else if (dir == 2) y++;  // South
		else x--;  // West

		// Keep track of min/max x/y for normalisation later
		if (y < min_y) min_y = y;
		if (y > max_y) max_y = y;
		if (x < min_x) min_x = x;
		if (x > max_x) max_x = x;
	}

	// 2. Normalise the walk coords

	for (auto& [x, y] : walk_coords) {
		x -= min_y;
		y -= min_x;
	}

	std::cout << "Generating image...\n";

	// 3. Count coord visit frequencies

	std::map<std::pair<int, int>, int> visit_counts;
	int max_visit_count = 0;
	for (const auto& coord : walk_coords) {
		visit_counts[coord]++;
		if (visit_counts[coord] > max_visit_count)
			max_visit_count = visit_counts[coord];
	}

	// 4. Create image based on normalised coords (making frequently visited coords brighter)

	int path_width = max_x - min_x + 1;
	int path_height = max_y - min_y + 1;
	cv::Mat img(path_width + 2 * BORDER, path_height + 2 * BORDER, CV_8UC3, cv::Scalar(0, 0, 0));

	for (const auto& coord : walk_coords) {
		int b = int(float(visit_counts[coord]) / max_visit_count * 235 + 20);  // Brightness (20 - 255)
		img.at<cv::Vec3b>(coord.second + BORDER, coord.first + BORDER) = cv::Vec3b(b, b, b);
	}
	img.at<cv::Vec3b>(walk_coords[0].second + BORDER, walk_coords[1].first + BORDER) = cv::Vec3b(255, 0, 0);
	cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
	cv::imwrite("./rand_walk.png", img);
	std::cout << "Image saved at ./rand_walk.png";

	return 0;
}
