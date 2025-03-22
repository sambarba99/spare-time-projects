/*
Barnsley fern generator

Author: Sam Barba
Created 26/10/2025
*/

#include <opencv2/opencv.hpp>
#include <random>

using std::vector;


using Transform = std::function<std::pair<double, double>(double, double)>;

const Transform F1 = [](double x, double y) {
    return std::make_pair(0.0, 0.16 * y);
};
const Transform F2 = [](double x, double y) {
    return std::make_pair(
		0.85 * x + 0.04 * y,
		-0.04 * x + 0.85 * y + 1.6
	);
};
const Transform F3 = [](double x, double y) {
    return std::make_pair(
		0.2 * x - 0.26 * y,
		0.23 * x + 0.22 * y + 1.6
	);
};
const Transform F4 = [](double x, double y) {
    return std::make_pair(
		-0.15 * x + 0.28 * y,
		0.26 * x + 0.24 * y + 0.44
	);
};

const vector<Transform> TRANSFORMS = {F1, F2, F3, F4};
const vector<double> PROBS = {0.01, 0.85, 0.07, 0.07};
const int NUM_STEPS = 1e6;
const int IMG_SIZE = 1500;


int main() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> dist(PROBS.begin(), PROBS.end());

	vector<std::pair<double, double>> points(NUM_STEPS + 1);
	points.emplace_back(0.0, 0.0);  // Start at origin

	double min_x = std::numeric_limits<double>::max(), max_x = std::numeric_limits<double>::min();
	double min_y = min_x, max_y = max_x;
	for (int i = 0; i < NUM_STEPS; i++) {
		int idx = dist(gen);
		auto [x, y] = points.back();
		auto [next_x, next_y] = TRANSFORMS[idx](x, y);
		points.emplace_back(next_x, next_y);
		
		// Keep track of min/max x/y for normalisation later
		if (y < min_y) min_y = y;
		if (y > max_y) max_y = y;
		if (x < min_x) min_x = x;
		if (x > max_x) max_x = x;
	}

	double size_x = max_x - min_x;
	double size_y = max_y - min_y;
	double scale = (IMG_SIZE - 1) / std::max(size_x, size_y);
	vector<cv::Point> scaled(points.size());
	for (const auto& p : points) {
		int sx = static_cast<int>((p.first - min_x) * scale);
		int sy = static_cast<int>((p.second - min_y) * scale);
		scaled.emplace_back(sx, sy);
	}

	int max_sx = INT_MIN, max_sy = INT_MIN;
	for (const auto& p : scaled) {
		if (p.x > max_sx) max_sx = p.x;
		if (p.y > max_sy) max_sy = p.y;
	}

	cv::Mat img(max_sy + 1, max_sx + 1, CV_8UC3, cv::Scalar(255, 255, 255));
	for (const auto& p : scaled)
		img.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(0, 128, 0);  // Set fern points to green
	cv::flip(img, img, 0);  // Flip (y-axis)
	cv::imwrite("./fern.png", img);
	std::cout << "Image saved at ./fern.png";

	return 0;
}
