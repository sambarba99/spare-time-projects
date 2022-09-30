"""
Tree class for binary_tree_demo.py

Author: Sam Barba
Created 08/09/2021
"""

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
		elif not data:
			tree = None
		else:
			tree = Tree(data)
		return tree

	def to_tuple(self):
		if not self:
			return None
		elif not self.left_child and not self.right_child:  # If leaf node
			return self.key
		else:
			return Tree.to_tuple(self.left_child), self.key, Tree.to_tuple(self.right_child)

	def list_data(self):
		if not self:
			return []
		return Tree.list_data(self.left_child) + [self.key] + Tree.list_data(self.right_child)

	def get_height(self):
		if not self:
			return 0
		return max(Tree.get_height(self.left_child), Tree.get_height(self.right_child)) + 1

	def in_order_traversal(self):
		if not self: return []

		return Tree.in_order_traversal(self.left_child) \
			+ [self.key] \
			+ Tree.in_order_traversal(self.right_child)

	def pre_order_traversal(self):
		if not self: return []

		return [self.key] \
			+ Tree.pre_order_traversal(self.left_child) \
			+ Tree.pre_order_traversal(self.right_child)

	def post_order_traversal(self):
		if not self: return []

		return Tree.post_order_traversal(self.left_child) \
			+ Tree.post_order_traversal(self.right_child) \
			+ [self.key]

	def bfs_traversal(self):
		"""Breadth-first search"""

		traversal = []
		queue = [self]

		while queue:
			node = queue.pop(0)
			traversal.append(node.key)

			if node.left_child:
				queue.append(node.left_child)
			if node.right_child:
				queue.append(node.right_child)

		return traversal

	def is_bst(self):
		def remove_none(*nums):
			return [x for x in nums if x is not None]

		if not self: return True, None, None

		is_left_bst, min_left, max_left = Tree.is_bst(self.left_child)
		is_right_bst, min_right, max_right = Tree.is_bst(self.right_child)

		is_tree_bst = is_left_bst and is_right_bst \
			and (max_left is None or self.key > max_left) \
			and (min_right is None or self.key < min_right)

		min_key = min(remove_none(min_left, self.key, min_right))
		max_key = max(remove_none(max_left, self.key, max_right))

		return is_tree_bst, min_key, max_key

	def is_balanced(self):
		if not self: return True

		left_height = Tree.get_height(self.left_child)
		right_height = Tree.get_height(self.right_child)

		return abs(left_height - right_height) <= 1 \
			and Tree.is_balanced(self.left_child) \
			and Tree.is_balanced(self.right_child)

	def insert(self, new_key):
		if self.key == new_key:
			return  # No duplicates allowed
		elif new_key > self.key:
			if self.right_child:
				self.right_child.insert(new_key)
			else:
				self.right_child = Tree(new_key)
		else:
			if self.left_child:
				self.left_child.insert(new_key)
			else:
				self.left_child = Tree(new_key)
