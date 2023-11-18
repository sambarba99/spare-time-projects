#ifndef NODE
#define NODE

#include <cmath>

class Node {
	public:
		int idx;
		float x;
		float y;
		float xVel;
		float yVel;

		Node(const int idx, const float x, const float y, const float xVel, const float yVel) {
			this->idx = idx;
			this->x = x;
			this->y = y;
			this->xVel = xVel;
			this->yVel = yVel;
		}

		bool operator==(const Node* other) {
 		   return this->idx == other->idx;
		}

		float euclideanDist(const Node* other) {
			return sqrt(pow(this->x - other->x, 2) + pow(this->y - other->y, 2));
		}
};

#endif
