/*
Minimum Spanning Tree demo

Controls:
	Left-click: add a node
	Right-click: reset graph

Author: Sam Barba
Created 15/11/2022
*/

#include <random>
#include <SFML/Graphics.hpp>

using std::vector;


const int SIZE = 600;
const int MAX_POINTS = 30;
const float MAX_VEL_MAGNITUDE = 1.f;
const float POINT_RADIUS = 5.f;
const int FPS = 60;

class Node {
	public:
		int idx;
		sf::Vector2f pos;
		sf::Vector2f vel;

		Node(const int idx, const sf::Vector2f& pos, const sf::Vector2f& vel) {
			this->idx = idx;
			this->pos = pos;
			this->vel = vel;
		}

		bool operator==(const Node& other) {
			return idx == other.idx;
		}

		float euclideanDist(const Node& other) {
			sf::Vector2f deltaPos = pos - other.pos;
			return sqrt(deltaPos.x * deltaPos.x + deltaPos.y * deltaPos.y);
		}
};

vector<Node> graph;
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> velDist(-MAX_VEL_MAGNITUDE, MAX_VEL_MAGNITUDE);
sf::RenderWindow window(sf::VideoMode(SIZE, SIZE), "Minimum Spanning Tree", sf::Style::Close);


vector<int> mst() {
	// Prim's algorithm

	vector<Node> outTree = graph;  // Initially set all nodes as out of tree
	vector<Node> inTree;
	vector<int> mstParents(graph.size(), -1);

	inTree.push_back(outTree[0]);  // Node 0 (arbitrary start) is first in tree
	outTree.erase(outTree.begin());
	float minDist, dist;

	while (!outTree.empty()) {
		Node nearestIn = inTree[0];
		Node nearestOut = outTree[0];
		minDist = nearestIn.euclideanDist(nearestOut);

		// Find nearest outside node to tree
		for (Node& nodeOut : outTree) {
			for (const Node& nodeIn : inTree) {
				dist = nodeOut.euclideanDist(nodeIn);
				if (dist < minDist) {
					minDist = dist;
					nearestOut = nodeOut;
					nearestIn = nodeIn;
				}
			}
		}

		mstParents[nearestOut.idx] = nearestIn.idx;
		inTree.push_back(nearestOut);
		outTree.erase(find(outTree.begin(), outTree.end(), nearestOut));
	}

	return mstParents;
}


void drawMST() {
	if (graph.empty()) return;

	window.clear(sf::Color(20, 20, 20));
	vector<int> mstParents = mst();

	for (int i = 1; i < graph.size(); i++) {  // Start from 1 because mstParents[0] = -1
		sf::Vertex line[] = {
			sf::Vertex(graph[i].pos),
			sf::Vertex(graph[mstParents[i]].pos)
		};
		window.draw(line, 2, sf::Lines);
	}

	for (const Node& node : graph) {
		sf::CircleShape circle(POINT_RADIUS);
		circle.setPosition(node.pos.x - POINT_RADIUS, node.pos.y - POINT_RADIUS);
		circle.setFillColor(sf::Color(230, 20, 20));
		window.draw(circle);
	}

	window.display();
}


void movePoints() {
	for (Node& node : graph) {
		node.pos += node.vel;

		if (node.pos.x < POINT_RADIUS) {
			node.vel.x *= -1;
			node.pos.x = POINT_RADIUS;
		} else if (node.pos.x > SIZE - POINT_RADIUS) {
			node.vel.x *= -1;
			node.pos.x = SIZE - POINT_RADIUS;
		}
		if (node.pos.y < POINT_RADIUS) {
			node.vel.y *= -1;
			node.pos.y = POINT_RADIUS;
		} else if (node.pos.y > SIZE - POINT_RADIUS) {
			node.vel.y *= -1;
			node.pos.y = SIZE - POINT_RADIUS;
		}
	}
}


int main() {
	window.setFramerateLimit(FPS);
	window.clear(sf::Color(20, 20, 20));
	window.display();
	sf::Event event;

	while (window.isOpen()) {
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::MouseButtonPressed:
					if (event.mouseButton.button == sf::Mouse::Left) {
						if (graph.size() < MAX_POINTS) {
							sf::Vector2f pos(sf::Mouse::getPosition(window));
							sf::Vector2f vel(velDist(gen), velDist(gen));
							graph.push_back(Node(graph.size(), pos, vel));
						}
					} else if (event.mouseButton.button == sf::Mouse::Right) {
						graph.clear();
						window.clear(sf::Color(20, 20, 20));
						window.display();
					}
					break;
			}
		}

		movePoints();
		drawMST();
	}

	return 0;
}
