"""
Tree class

Author: Sam Barba
Created 08/09/2021
"""

class Tree:
	def __init__(self, data):
		self.data = data
		self.left_child = None
		self.right_child = None


	def insert(self, new_data):
		if self.data == new_data:
			return  # No duplicates allowed
		elif new_data < self.data:
			if self.left_child:
				self.left_child.insert(new_data)
			else:
				self.left_child = Tree(new_data)
		else:
			if self.right_child:
				self.right_child.insert(new_data)
			else:
				self.right_child = Tree(new_data)


	def to_tuple(self):
		if not self:
			return None
		elif not self.left_child and not self.right_child:  # If leaf node
			return self.data
		else:
			return Tree.to_tuple(self.left_child), self.data, Tree.to_tuple(self.right_child)


	def list_data(self):
		if not self:
			return []
		return Tree.list_data(self.left_child) + [self.data] + Tree.list_data(self.right_child)


	def get_height(self):
		if not self:
			return 0
		return max(Tree.get_height(self.left_child), Tree.get_height(self.right_child)) + 1


	def in_order_traversal(self):
		if not self: return []

		return Tree.in_order_traversal(self.left_child) \
			+ [self.data] \
			+ Tree.in_order_traversal(self.right_child)


	def pre_order_traversal(self):
		if not self: return []

		return [self.data] \
			+ Tree.pre_order_traversal(self.left_child) \
			+ Tree.pre_order_traversal(self.right_child)


	def post_order_traversal(self):
		if not self: return []

		return Tree.post_order_traversal(self.left_child) \
			+ Tree.post_order_traversal(self.right_child) \
			+ [self.data]


	def breadth_first_traversal(self):
		traversal = []
		queue = [self]

		while queue:
			node = queue.pop(0)
			traversal.append(node.data)

			if node.left_child:
				queue.append(node.left_child)
			if node.right_child:
				queue.append(node.right_child)

		return traversal


	def is_bst(self, prev='0'):
		"""Is Binary Search Tree"""

		if self:
			if not Tree.is_bst(self.left_child, prev):
				return False

			# Handle equal-valued nodes
			if self.data < prev:
				return False

			return Tree.is_bst(self.right_child)

		return True


	def is_balanced(self):
		if not self: return True

		left_height = Tree.get_height(self.left_child)
		right_height = Tree.get_height(self.right_child)

		return abs(left_height - right_height) <= 1 \
			and Tree.is_balanced(self.left_child) \
			and Tree.is_balanced(self.right_child)
