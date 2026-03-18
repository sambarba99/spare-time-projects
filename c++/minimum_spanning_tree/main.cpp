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

		float euclidean_dist(const Node& other) {
			sf::Vector2f delta = pos - other.pos;
			return delta.x * delta.x + delta.y * delta.y;
		}
};

vector<Node> graph;
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> vel_dist(-MAX_VEL_MAGNITUDE, MAX_VEL_MAGNITUDE);
sf::RenderWindow window(sf::VideoMode(SIZE, SIZE), "Minimum Spanning Tree", sf::Style::Close);


vector<int> mst() {
	// Prim's algorithm

	vector<Node> out_tree = graph;  // Initially set all nodes as out of tree
	vector<Node> in_tree;
	vector<int> mst_parents(graph.size(), -1);

	in_tree.emplace_back(out_tree[0]);  // Node 0 (arbitrary start) is first in tree
	out_tree.erase(out_tree.begin());
	float min_dist, dist;

	while (!out_tree.empty()) {
		Node nearest_in = in_tree[0];
		Node nearest_out = out_tree[0];
		min_dist = nearest_in.euclidean_dist(nearest_out);

		// Find nearest outside node to tree
		for (Node& node_out : out_tree) {
			for (const Node& node_in : in_tree) {
				dist = node_out.euclidean_dist(node_in);
				if (dist < min_dist) {
					min_dist = dist;
					nearest_out = node_out;
					nearest_in = node_in;
				}
			}
		}

		mst_parents[nearest_out.idx] = nearest_in.idx;
		in_tree.emplace_back(nearest_out);
		out_tree.erase(find(out_tree.begin(), out_tree.end(), nearest_out));
	}

	return mst_parents;
}


void draw_mst() {
	if (graph.empty())
		return;

	window.clear(sf::Color(20, 20, 20));
	vector<int> mst_parents = mst();

	for (int i = 1; i < graph.size(); i++) {  // Start from 1 because mst_parents[0] = -1
		sf::Vertex line[] = {
			sf::Vertex(graph[i].pos),
			sf::Vertex(graph[mst_parents[i]].pos)
		};
		window.draw(line, 2, sf::Lines);
	}

	sf::CircleShape point(POINT_RADIUS);
	point.setOrigin(POINT_RADIUS, POINT_RADIUS);
	point.setFillColor(sf::Color(230, 20, 20));

	for (const Node& node : graph) {
		point.setPosition(node.pos.x, node.pos.y);
		window.draw(point);
	}

	window.display();
}


void move_points() {
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
							sf::Vector2f vel(vel_dist(gen), vel_dist(gen));
							graph.emplace_back(Node(graph.size(), pos, vel));
						}
					} else if (event.mouseButton.button == sf::Mouse::Right) {
						graph.clear();
						window.clear(sf::Color(20, 20, 20));
						window.display();
					}
					break;
			}
		}

		move_points();
		draw_mst();
	}

	return 0;
}
