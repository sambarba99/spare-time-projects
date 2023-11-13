"""
Huffman coding demo

Author: Sam Barba
Created 22/11/2022
"""

import heapq

from huffman_tree import HuffmanTree
from tree_plotter import plot_tree


# STRING = 'a mad boxer shot a quick, gloved jab to the jaw of his dizzy opponent'
# STRING = 'pack my box with five dozen liquor jugs'
# STRING = 'sphinx of black quartz, judge my vow'
# STRING = 'the five boxing wizards jump quickly'
# STRING = 'the quick brown fox jumps over the lazy dog'
# STRING = 'we promptly judged antique ivory buckles for the next prize'


if __name__ == '__main__':
	with open('lyrics.txt', 'r') as file:
		STRING = file.read()[:-1].replace('\n', ';')

	assert '_' not in STRING, "Cannot have '_' char in string due to tree rendering in tree_plotter.py"

	print(f'\nOriginal string:\n{STRING}')

	# Get frequencies of each char
	set_str = set(STRING)
	freqs = [STRING.count(c) for c in set_str]

	# Convert to probabilities/weights (so they sum to 1)
	sum_ = sum(freqs)
	weights = [f / sum_ for f in freqs]
	symbols_weights = list(zip(set_str, weights))

	# Sort symbols/weights (not needed - just for visualisation)
	symbols_weights.sort(key=lambda tup: (-tup[1], tup[0]))
	symbols_weights = dict(symbols_weights)
	print(f'\nSymbols and weights:')
	print(*symbols_weights.items(), sep='\n')

	# Convert to Huffman tree nodes
	nodes = []
	for s, w in symbols_weights.items():
		heapq.heappush(nodes, HuffmanTree(s, w))

	while len(nodes) > 1:
		left = heapq.heappop(nodes)
		right = heapq.heappop(nodes)
		new_node = HuffmanTree(left.symbol + right.symbol, left.weight + right.weight, left, right)
		heapq.heappush(nodes, new_node)

	tree = nodes[0]

	huffman_dict = dict()
	tree.create_huffman_dict(huffman_dict)
	huffman_dict = {c: huffman_dict[c] for c in symbols_weights}
	print(f'\nHuffman codes:\n{huffman_dict}')

	binary = ''.join(format(ord(c), 'b') for c in STRING)
	print(f'\nString in binary:\n{binary}')

	encoded = tree.encode(STRING, huffman_dict)
	print(f'\nHuffman coded:\n{encoded}')

	compression_amount = 1 - (len(encoded) / len(binary))
	print(f'\nCompression %: {(compression_amount * 100):.2f}')

	print(f'\nDecoded:\n{tree.decode(encoded)}')

	plot_tree(tree)
