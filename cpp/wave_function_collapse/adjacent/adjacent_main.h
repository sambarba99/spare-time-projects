/*
Visualisation of Wave Function Collapse (adjacent tiling algorithm)

1. First, a set of tile images is loaded from ./tile_imgs (change via COLLAGE_TYPE in ../main.cpp).
2. Tile objects are created using these images, together with predefined edge colour codes (north/east/south/west edges,
read clockwise) and the number of possible orientations of the tile.
3. Adjacency rules are created by using the edge colour codes of the tiles.
4. A grid of cells (../cell.h) is then initiated, each one initially having multiple possible states (tiles). This is
their superposition. By iterating WFC and checking which neighbour tile states are allowed, each cell will eventually
have just one state (its superposition is 'collapsed'), meaning its tile image can be visualised.

Author: Sam Barba
Created 19/07/2023
*/

#include <filesystem>
#include <SFML/Graphics.hpp>

#include "../cell.h"

using std::string;
using std::vector;


const int COLLAGE_WIDTH = 49;  // In tiles
const int COLLAGE_HEIGHT = 29;
const bool RENDER_ENTROPY_VALUES = true;
const int FPS = 60;

class Tile {
	public:
		sf::Texture texture;
		string northEdge, eastEdge, southEdge, westEdge;
		std::map<string, vector<int>> neighbourOptions;

		Tile(const sf::Texture& texture, const vector<string>& edgeColourCodes) {
			this->texture = texture;
			northEdge = edgeColourCodes[0];
			eastEdge = edgeColourCodes[0];
			southEdge = edgeColourCodes[0];
			westEdge = edgeColourCodes[0];
			neighbourOptions["north"] = {};
			neighbourOptions["east"] = {};
			neighbourOptions["south"] = {};
			neighbourOptions["west"] = {};
		}

		void generateAdjacencyRules(const vector<Tile>& tiles) {
			for (int idx = 0; idx < tiles.size(); ++idx) {
				Tile other = tiles[idx];
				if (northEdge == string(other.southEdge.rbegin(), other.southEdge.rend()))
					neighbourOptions["north"].push_back(idx);
				if (eastEdge == string(other.westEdge.rbegin(), other.westEdge.rend()))
					neighbourOptions["east"].push_back(idx);
				if (southEdge == string(other.northEdge.rbegin(), other.northEdge.rend()))
					neighbourOptions["south"].push_back(idx);
				if (westEdge == string(other.eastEdge.rbegin(), other.eastEdge.rend()))
					neighbourOptions["west"].push_back(idx);
			}
		}
};

int tileSize;
vector<Tile> tiles;
sf::Event event;
sf::Font font;


void generateTiles(const string collageType) {
	// Create Tile objects by reading image files

	tiles.clear();
	string folderPath = "./adjacent/tile_imgs/" + collageType;

	for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
		if (entry.path().extension() == ".png") {
			string filename = entry.path().filename().string();
			std::istringstream ss(filename);
			vector<string> parts;
			string part;

			while (std::getline(ss, part, '_'))
				parts.push_back(part);

			vector<string> edgeColourCodes(parts.end() - 5, parts.end() - 1);
			int numOrientations = std::stoi(parts.back().substr(0, parts.back().find('.')));

			sf::Texture texture;
			texture.loadFromFile(entry.path().string());
			tiles.push_back(Tile(texture, edgeColourCodes));

			if (numOrientations > 1) {
				vector<string> oldEdges = edgeColourCodes;
				for (int n = 1; n < numOrientations; ++n) {
					sf::RenderTexture renderTexture;
					renderTexture.create(texture.getSize().x, texture.getSize().y);
					sf::Sprite sprite(texture);
					sprite.setOrigin(texture.getSize().x / 2, texture.getSize().y / 2);
					sprite.setPosition(texture.getSize().y / 2, texture.getSize().x / 2);  // Center it
					sprite.setRotation(n * 90);  // Rotate by n * 90 degrees
					renderTexture.draw(sprite);
					renderTexture.display();

					sf::Texture rotatedTexture = renderTexture.getTexture();
					vector<string> newEdges(4);
					for (int i = 0; i < 4; i++)
						newEdges[i] = oldEdges[(i - n + 4) % 4];

					tiles.push_back(Tile(rotatedTexture, newEdges));
				}
			}
		}
	}

	for (Tile& tile : tiles)
		tile.generateAdjacencyRules(tiles);
}


string waveFunctionCollapse(vector<vector<Cell>>& grid, const bool firstCell = false) {
	for (vector<Cell>& row : grid)
		for (Cell& cell : row)
			cell.observe();

	return "all cells collapsed";
}


void draw(sf::RenderWindow& window, vector<vector<Cell>>& grid) {
	window.clear();

	for (vector<Cell>& row : grid)
		for (Cell& cell : row) {
			if (cell.isCollapsed()) {
				int tileIdx = cell.tileOptions[0];
				sf::Sprite sprite;
				sprite.setTexture(tiles[tileIdx].texture);
				sprite.setPosition(cell.x * tileSize, cell.y * tileSize);
				window.draw(sprite);
			} else if (RENDER_ENTROPY_VALUES) {
				sf::Text cellText(std::to_string(cell.entropy()), font, 11);
				cellText.setPosition(int(cell.x * tileSize), int(cell.y * tileSize));
				cellText.setFillColor(sf::Color::White);
				window.draw(cellText);
			}
		}

	window.display();

	while (window.isOpen())
		while (window.pollEvent(event))
			if (event.type == sf::Event::Closed)
				window.close();
}


int adjacent_tiling(const string collageType) {
	generateTiles(collageType);
	tileSize = tiles[0].texture.getSize().x;
	std::uniform_int_distribution<int> dist(0, tiles.size() - 1);

	sf::RenderWindow window(
		sf::VideoMode(COLLAGE_WIDTH * tileSize, COLLAGE_HEIGHT * tileSize),
		"Wave Function Collapse (adjacent tiling algorithm)"
	);
	window.setFramerateLimit(FPS);
	font.loadFromFile("C:/Windows/Fonts/consola.ttf");

	while (true) {
		vector<vector<Cell>> grid(COLLAGE_HEIGHT, vector<Cell>(COLLAGE_WIDTH));
		for (int y = 0; y < COLLAGE_HEIGHT; y++)
			for (int x = 0; x < COLLAGE_WIDTH; x++)
				grid[y][x] = Cell(y, x, tiles.size());

		draw(window, grid);
		sf::sleep(sf::milliseconds(1000));
		waveFunctionCollapse(grid, true);
		while (true) {
			string result = waveFunctionCollapse(grid);
			if (result == "contradiction" || result == "all cells collapsed")
				break;
		}
		// If result == "all cells collapsed", screenshot to ../<collageType>.png
		sf::sleep(sf::milliseconds(2000));
	}

	return 0;
}
