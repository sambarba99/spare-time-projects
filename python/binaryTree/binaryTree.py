# Binary Tree demo
# Author: Sam Barba
# Created 08/09/2021

import random

# ---------------------------------------------------------------------------------------------------- #
# ---------------------------------------------  CLASSES  -------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

class Tree:
	# Without 'value' variable for sake of demo
	def __init__(self, key):
		self.key = key
		self.left_child = None
		self.right_child = None

	@staticmethod
	def parse_tuple(data):
		if isinstance(data, tuple) and len(data) == 3:
			tree = Tree(data[1])
			tree.left_child = Tree.parse_tuple(data[0])
			tree.right_child = Tree.parse_tuple(data[2])
		elif data is None:
			tree = None
		else:
			tree = Tree(data)
		return tree

	def to_tuple(self):
		if self is None:
			return None
		elif self.left_child is None and self.right_child is None:  # If leaf node
			return self.key
		else:
			return Tree.to_tuple(self.left_child), self.key, Tree.to_tuple(self.right_child)

	def list_data(self):
		if self is None:
			return []
		return Tree.list_data(self.left_child) + [self.key] + Tree.list_data(self.right_child)

	def get_height(self):
		if self is None:
			return 0
		return max(Tree.get_height(self.left_child), Tree.get_height(self.right_child)) + 1

	def in_order_traversal(self):
		if self is None: return []

		return (Tree.in_order_traversal(self.left_child)
			+ [self.key]
			+ Tree.in_order_traversal(self.right_child))

	# Depth-first search
	def pre_order_traversal(self):
		if self is None: return []

		return ([self.key]
			+ Tree.pre_order_traversal(self.left_child)
			+ Tree.pre_order_traversal(self.right_child))

	def post_order_traversal(self):
		if self is None: return []

		return (Tree.in_order_traversal(self.left_child)
			+ Tree.in_order_traversal(self.right_child)
			+ [self.key])

	def is_bst(self):
		if self is None: return True, None, None

		is_left_bst, min_left, max_left = Tree.is_bst(self.left_child)
		is_right_bst, min_right, max_right = Tree.is_bst(self.right_child)

		is_tree_bst = (is_left_bst and is_right_bst and
			(max_left is None or self.key > max_left) and
			(min_right is None or self.key < min_right))

		min_key = min(Tree.__remove_none(min_left, self.key, min_right))
		max_key = max(Tree.__remove_none(max_left, self.key, max_right))

		return is_tree_bst, min_key, max_key

	def __remove_none(*nums):
		return [x for x in nums if x is not None]

	def is_balanced(self):
		if self is None: return True

		left_height = Tree.get_height(self.left_child)
		right_height = Tree.get_height(self.right_child)

		return ((abs(left_height - right_height) <= 1)
			and Tree.is_balanced(self.left_child)
			and Tree.is_balanced(self.right_child))

	def insert(self, new_key):
		if self.key == new_key:
			return  # No duplicates allowed
		elif new_key > self.key:
			if self.right_child is None:
				self.right_child = Tree(new_key)
			else:
				self.right_child.insert(new_key)
		else:
			if self.left_child is None:
				self.left_child = Tree(new_key)
			else:
				self.left_child.insert(new_key)

	def display(self):
		lines, *_ = self.__display_aux()
		for line in lines:
			print(line)
	
	def __display_aux(self):
		# No child
		if self.right_child is None and self.left_child is None:
			line = str(self.key)
			width = len(line)
			height = 1
			middle = width // 2
			return [line], width, height, middle

		# Only left child
		if self.right_child is None:
			lines, n, p, x = self.left_child.__display_aux()
			s = str(self.key)
			u = len(s)
			first_line = (x + 1) * " " + (n - x - 1) * "_" + s
			second_line = x * " " + "/" + (n - x - 1 + u) * " "
			shifted_lines = [line + u * " " for line in lines]
			return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

		# Only right child
		if self.left_child is None:
			lines, n, p, x = self.right_child.__display_aux()
			s = str(self.key)
			u = len(s)
			first_line = s + x * "_" + (n - x) * " "
			second_line = (u + x) * " " + "\\" + (n - x - 1) * " "
			shifted_lines = [u * " " + line for line in lines]
			return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

		# Two children
		left, n, p, x = self.left_child.__display_aux()
		right, m, q, y = self.right_child.__display_aux()
		s = str(self.key)
		u = len(s)
		first_line = (x + 1) * " " + (n - x - 1) * "_" + s + y * "_" + (m - y) * " "
		second_line = x * " " + "/" + (n - x - 1 + u + y) * " " + "\\" + (m - y - 1) * " "
		if p < q:
			left += [n * " "] * (q - p)
		elif p > q:
			right += [m * " "] * (p - q)
		lines = [first_line, second_line] + [a + u * " " + b for a, b in zip(left, right)]
		return lines, n + m + u, max(p, q) + 2, n + u // 2

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def make_random_binary_tree():
	with open("peopleNames.txt", "r") as file:
		names = file.readlines()

	names = [name.strip("\n") for name in names]

	tree_keys = random.sample(names, 15)

	bin_tree = Tree(tree_keys.pop())

	for name in tree_keys:
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

while True:
	binary_tree = make_random_binary_tree()
	while binary_tree.is_balanced():
		binary_tree = make_random_binary_tree()

	print("{:>21}:  {}".format("Tree", binary_tree.to_tuple()))
	print("{:>21}:  {}".format("Tree height", binary_tree.get_height()))
	print("{:>21}:  {}".format("Is Binary Search Tree", binary_tree.is_bst()[0]))
	print("{:>21}:  {}".format("Is balanced", binary_tree.is_balanced()), "\n")

	binary_tree.display()

	binary_tree = make_balanced_bst(binary_tree.list_data())

	print("\n{:-^50}\n".format(" After balancing "))
	print("{:>21}:  {}".format("Tree", binary_tree.to_tuple()))
	print("{:>21}:  {}".format("Tree height", binary_tree.get_height()))
	print("{:>21}:  {}".format("In-order traversal", binary_tree.in_order_traversal()))
	print("{:>21}:  {}".format("Pre-order traversal", binary_tree.pre_order_traversal()))
	print("{:>21}:  {}".format("Post-order traversal", binary_tree.post_order_traversal()), "\n")

	binary_tree.display()

	choice = input("\nEnter to continue or X to exit: ").upper()
	if choice and choice[0] == 'X':
		break
	print()
