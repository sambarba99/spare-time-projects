/*
Minimum Spanning Tree demo

Author: Sam Barba
Created 15/11/2022

Controls:
Left-click: add a node
Right-click: reset graph
*/

#include <algorithm>
#include <SFML/Graphics.hpp>
#include <vector>

#include "node.h"

using std::find;
using std::remove;
using std::vector;

const int SIZE = 600;

vector<Node*> graph;
sf::RenderWindow window(sf::VideoMode(SIZE, SIZE), "Minimum Spanning Tree", sf::Style::Close);

vector<int> mst() {
	// Prim's algorithm

	vector<Node*> outTree = graph;  // Initially set all nodes as out of tree
	vector<Node*> inTree;
	vector<int> mstParents(graph.size(), -1);

	inTree.push_back(outTree[0]);  // Node 0 (arbitrary start) is first in tree
	outTree.erase(outTree.begin());
	float minDist, dist;

	while (!outTree.empty()) {
		Node* nearestIn = inTree[0];
		Node* nearestOut = outTree[0];
		minDist = nearestIn->euclideanDist(nearestOut);

		// Find nearest outside node to tree
		for (Node* nodeOut : outTree) {
			for (Node* nodeIn : inTree) {
				dist = nodeOut->euclideanDist(nodeIn);
				if (dist < minDist) {
					minDist = dist;
					nearestOut = nodeOut;
					nearestIn = nodeIn;
				}
			}
		}

		mstParents[nearestOut->idx] = nearestIn->idx;
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
		float startX = graph[i]->x, startY = graph[i]->y;
		float endX = graph[mstParents[i]]->x, endY = graph[mstParents[i]]->y;
		sf::Vertex line[] = {
			sf::Vertex(sf::Vector2f(startX, startY)),
			sf::Vertex(sf::Vector2f(endX, endY))
		};
		window.draw(line, 2, sf::Lines);
	}

	for (Node* node : graph) {
		sf::CircleShape circle(5.f);
		circle.setPosition(node->x - 2.5f, node->y - 2.5f);
		circle.setFillColor(sf::Color(230, 20, 20));
		window.draw(circle);
	}

	window.display();
}

void movePoints() {
	for (Node* node : graph) {
		node->x += node->xVel;
		node->y += node->yVel;

		while (node->x < 5 || node->x > SIZE - 5) {
			node->xVel *= -1.f;
			node->x += node->xVel;
		}
		while (node->y < 5 || node->y > SIZE - 5) {
			node->yVel *= -1.f;
			node->y += node->yVel;
		}
	}
}

float randomFloat(const float a, const float b) {
    float diff = b - a;
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return a + diff * r;
}

int main() {
	window.clear(sf::Color(20, 20, 20));
	window.display();

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			switch (event.type) {
				case sf::Event::Closed:
					window.close();
					break;
				case sf::Event::MouseButtonPressed:
					if (event.mouseButton.button == sf::Mouse::Left) {
						if (graph.size() < 30) {  // Stay within size limit
							sf::Vector2i mousePos = sf::Mouse::getPosition(window);
							int mouseX = mousePos.x, mouseY = mousePos.y;
							float xVel = randomFloat(-0.02f, 0.02f);
							float yVel = randomFloat(-0.02f, 0.02f);
							Node* node = new Node(graph.size(), mouseX, mouseY, xVel, yVel);
							graph.push_back(node);
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
