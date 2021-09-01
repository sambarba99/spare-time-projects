"""
Binary Tree demo

Author: Sam Barba
Created 08/09/2021
"""

from binary_tree import Tree
import numpy as np

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def make_random_binary_tree():
	names = np.genfromtxt('people_names.txt', dtype=str, delimiter='\n')

	tree_keys = np.random.choice(names, size=15, replace=False)

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
	while True:
		binary_tree = make_random_binary_tree()
		while binary_tree.is_balanced():
			binary_tree = make_random_binary_tree()

		print('{:>21}:  {}'.format('Tree', binary_tree.to_tuple()))
		print('{:>21}:  {}'.format('Tree height', binary_tree.get_height()))
		print('{:>21}:  {}'.format('Is Binary Search Tree', binary_tree.is_bst()[0]))
		print('{:>21}:  {}'.format('Is balanced', binary_tree.is_balanced()), '\n')

		binary_tree.display()

		binary_tree = make_balanced_bst(binary_tree.list_data())

		print('\n{:-^50}\n'.format(' After balancing '))
		print('{:>21}:  {}'.format('Tree', binary_tree.to_tuple()))
		print('{:>21}:  {}'.format('Tree height', binary_tree.get_height()))
		print('{:>21}:  {}'.format('In-order traversal', binary_tree.in_order_traversal()))
		print('{:>21}:  {}'.format('Pre-order traversal', binary_tree.pre_order_traversal()))
		print('{:>21}:  {}'.format('Post-order traversal', binary_tree.post_order_traversal()), '\n')

		binary_tree.display()

		choice = input('\nEnter to continue or X to exit\n>>> ').upper()
		if choice and choice[0] == 'X':
			break
		print()
