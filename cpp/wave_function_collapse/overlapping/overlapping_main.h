/*
Visualisation of Wave Function Collapse (overlapping tiling algorithm)

1. First, an input image from ./src_imgs is read (change via COLLAGE_TYPE in ../main.cpp).
2. Patches (tiles) of a predefined size are extracted from the image into a list. E.g. with a source image of size
20x20, this tile list would be 400 long.
3. Tile objects are created using these images, and adjacency rules are created by using the colours on the edges of
each tile (TILE_SIZE - 1 wide) and comparing the overlap with the edges of other tiles.
4. A grid of cells (../cell.h) is then initiated, each one initially having multiple possible states (tiles). This is
their superposition. By iterating WFC and checking which neighbour tile states are allowed, each cell will eventually
have just one state (its superposition is 'collapsed'), meaning its tile image can be visualised.

Author: Sam Barba
Created 15/02/2025
*/

#include <SFML/Graphics.hpp>
#include <vector>


int overlapping_tiling(const std::string collage_type) {
	sf::RenderWindow window(sf::VideoMode(800, 600), "SFML Image Rendering");
	sf::Event event;

	sf::Texture texture;
	texture.loadFromFile("./adjacent/tile_imgs/circuit/bridge_aba_aca_aba_aca_2.png");

	std::vector<sf::Sprite> tiles;
	for (int y = 0; y < window.getSize().y / 20; y++) {
		for (int x = 0; x < window.getSize().x / 20; x++) {
			sf::Sprite sprite;
			sprite.setTexture(texture);
			sprite.setPosition(x * 20, y * 20);
			tiles.push_back(sprite);
		}
	}

	while (window.isOpen()) {
		while (window.pollEvent(event))
			if (event.type == sf::Event::Closed)
				window.close();

		window.clear();
		for (const sf::Sprite& tile : tiles)
			window.draw(tile);
		window.display();
	}

	return 0;
}
