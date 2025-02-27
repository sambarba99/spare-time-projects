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

#include "node.h"

using std::vector;


const int SIZE = 600;
const int MAX_POINTS = 30;
const float MAX_VEL_MAGNITUDE = 1.f;
const float POINT_RADIUS = 5.f;
const int FPS = 60;

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
		float startX = graph[i].x, startY = graph[i].y;
		float endX = graph[mstParents[i]].x, endY = graph[mstParents[i]].y;
		sf::Vertex line[] = {
			sf::Vertex(sf::Vector2f(startX, startY)),
			sf::Vertex(sf::Vector2f(endX, endY))
		};
		window.draw(line, 2, sf::Lines);
	}

	for (const Node& node : graph) {
		sf::CircleShape circle(POINT_RADIUS);
		circle.setPosition(node.x - POINT_RADIUS, node.y - POINT_RADIUS);
		circle.setFillColor(sf::Color(230, 20, 20));
		window.draw(circle);
	}

	window.display();
}


void movePoints() {
	for (Node& node : graph) {
		node.x += node.xVel;
		node.y += node.yVel;

		while (node.x < POINT_RADIUS || node.x > SIZE - POINT_RADIUS) {
			node.xVel *= -1.f;
			node.x += node.xVel;
		}
		while (node.y < POINT_RADIUS || node.y > SIZE - POINT_RADIUS) {
			node.yVel *= -1.f;
			node.y += node.yVel;
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
							sf::Vector2i mousePos = sf::Mouse::getPosition(window);
							int mouseX = mousePos.x, mouseY = mousePos.y;
							float xVel = velDist(gen);
							float yVel = velDist(gen);
							graph.push_back(Node(graph.size(), mouseX, mouseY, xVel, yVel));
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
}
