"""
Tree class for Huffman coding demo

Author: Sam Barba
Created 22/11/2022
"""


class HuffmanTree:
	def __init__(self, symbol, weight, left_child=None, right_child=None):
		self.symbol = symbol
		self.weight = weight
		self.left_child = left_child
		self.right_child = right_child

	def __lt__(self, other):
		return self.weight < other.weight or (self.weight == other.weight and self.symbol < other.symbol)

	def create_huffman_dict(self, huffman_dict, h=''):
		if not self.left_child and not self.right_child:  # Leaf
			huffman_dict[self.symbol] = h

		if self.left_child:
			self.left_child.create_huffman_dict(huffman_dict, h + '0')
		if self.right_child:
			self.right_child.create_huffman_dict(huffman_dict, h + '1')

	def encode(self, string, huffman_dict=None):
		if not huffman_dict:
			huffman_dict = dict()
			self.create_huffman_dict(huffman_dict)

		return ''.join(huffman_dict[c] for c in string)

	def decode(self, encoded):
		is_binary = all(c in '01' for c in encoded)
		assert is_binary, f"Encoded string must be binary (got '{encoded}')"

		result = []
		index = 0
		while index < len(encoded) - 1:
			index = self.__decode_util(index, encoded, result)
		return ''.join(result)

	def __decode_util(self, index, encoded, result):
		if not self.left_child and not self.right_child:  # Leaf
			result.append(self.symbol)
			return index

		if encoded[index] == '0' and self.left_child:
			return self.left_child.__decode_util(index + 1, encoded, result)
		elif self.right_child:
			return self.right_child.__decode_util(index + 1, encoded, result)
