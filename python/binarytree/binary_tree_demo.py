"""
Binary Tree demo

Author: Sam Barba
Created 08/09/2021
"""

import numpy as np

from binary_tree import Tree
from tree_plotter import plot_tree

N_NODES = 31  # No. people names (max 113)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def make_random_binary_tree():
	with open('people_names.txt', 'r') as file:
		names = file.read().splitlines()

	tree_keys = np.random.choice(names, size=N_NODES, replace=False)

	bin_tree = Tree(tree_keys[0])

	for name in tree_keys[1:]:
		bin_tree.insert(name)

	return bin_tree

def make_balanced_bst(data, lo=0, hi=None):
	data.sort()

	if hi is None:
		hi = len(data) - 1
	if lo > hi:
		return None

	mid = (lo + hi) // 2

	root = Tree(data[mid])
	root.left_child = make_balanced_bst(data, lo, mid - 1)
	root.right_child = make_balanced_bst(data, mid + 1, hi)
	return root

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	assert isinstance(N_NODES, int) and 1 <= N_NODES <= 113

	binary_tree = make_random_binary_tree()
	while binary_tree.is_balanced() and binary_tree.get_height() > 1:
		binary_tree = make_random_binary_tree()

	print('Tree:\n', binary_tree.to_tuple())
	print('Height:', binary_tree.get_height())
	print('Is Binary Search Tree:', binary_tree.is_bst()[0])
	print('Balanced:', binary_tree.is_balanced())

	plot_tree(binary_tree, 'unbalanced')

	binary_tree = make_balanced_bst(binary_tree.list_data())

	print()
	print('-' * 50, 'After balancing', '-' * 50)
	print()
	print('Tree:\n', binary_tree.to_tuple())
	print('Height:', binary_tree.get_height())
	print('In-order traversal:\n', binary_tree.in_order_traversal())
	print('Pre-order traversal:\n', binary_tree.pre_order_traversal())
	print('Post-order traversal:\n', binary_tree.post_order_traversal())
	print('Breadth-first traversal:\n', binary_tree.bfs_traversal())

	plot_tree(binary_tree, 'balanced')
