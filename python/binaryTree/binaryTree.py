# Binary Tree demo
# Author: Sam Barba
# Created 08/09/2021

import random

# ---------------------------------------------------------------------------------------------------- #
# ---------------------------------------------  CLASSES  -------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

class Tree:
	# without 'value' variable for sake of demo
	def __init__(self, key):
		self.key = key
		self.leftChild = None
		self.rightChild = None

	@staticmethod
	def parseTuple(data):
		if isinstance(data, tuple) and len(data) == 3:
			tree = Tree(data[1])
			tree.leftChild = Tree.parseTuple(data[0])
			tree.rightChild = Tree.parseTuple(data[2])
		elif data == None:
			tree = None
		else:
			tree = Tree(data)
		return tree

	def toTuple(self):
		if self == None:
			return None
		elif self.leftChild == None and self.rightChild == None: # if leaf node
			return self.key
		else:
			return Tree.toTuple(self.leftChild), self.key, Tree.toTuple(self.rightChild)

	def listData(self):
		if self == None:
			return []
		return Tree.listData(self.leftChild) + [self.key] + Tree.listData(self.rightChild)

	def getHeight(self):
		if self == None:
			return 0
		return max(Tree.getHeight(self.leftChild), Tree.getHeight(self.rightChild)) + 1

	def inOrderTraversal(self):
		if self == None: return []

		return (Tree.inOrderTraversal(self.leftChild)
			+ [self.key]
			+ Tree.inOrderTraversal(self.rightChild))

	def preOrderTraversal(self): # depth-first search
		if self == None: return []

		return ([self.key]
			+ Tree.preOrderTraversal(self.leftChild)
			+ Tree.preOrderTraversal(self.rightChild))

	def postOrderTraversal(self):
		if self == None: return []

		return (Tree.inOrderTraversal(self.leftChild)
			+ Tree.inOrderTraversal(self.rightChild)
			+ [self.key])

	def isBST(self):
		if self == None: return True, None, None

		isLeftBST, minLeft, maxLeft = Tree.isBST(self.leftChild)
		isRightBST, minRight, maxRight = Tree.isBST(self.rightChild)

		isTreeBST = (isLeftBST and isRightBST and 
			(maxLeft == None or self.key > maxLeft) and 
			(minRight == None or self.key < minRight))

		minKey = min(Tree.__removeNone(minLeft, self.key, minRight))
		maxKey = max(Tree.__removeNone(maxLeft, self.key, maxRight))

		return isTreeBST, minKey, maxKey

	def __removeNone(*nums):
		return [x for x in nums if x != None]

	def isBalanced(self):
		if self == None: return True

		leftHeight = Tree.getHeight(self.leftChild)
		rightHeight = Tree.getHeight(self.rightChild)

		return ((abs(leftHeight - rightHeight) <= 1)
			and Tree.isBalanced(self.leftChild)
			and Tree.isBalanced(self.rightChild))

	def insert(self, newKey):
		if self.key == newKey:
			return # No duplicates allowed
		elif newKey > self.key:
			if self.rightChild == None:
				self.rightChild = Tree(newKey)
			else:
				self.rightChild.insert(newKey)
		else:
			if self.leftChild == None:
				self.leftChild = Tree(newKey)
			else:
				self.leftChild.insert(newKey)

	def display(self):
		lines, *_ = self.__displayAux()
		for line in lines:
			print(line)
	
	def __displayAux(self):
		# No child
		if self.rightChild == None and self.leftChild == None:
			line = str(self.key)
			width = len(line)
			height = 1
			middle = width // 2
			return [line], width, height, middle

		# Only left child
		if self.rightChild == None:
			lines, n, p, x = self.leftChild.__displayAux()
			s = str(self.key)
			u = len(s)
			firstLine = (x + 1) * " " + (n - x - 1) * "_" + s
			secondLine = x * " " + "/" + (n - x - 1 + u) * " "
			shiftedLines = [line + u * " " for line in lines]
			return [firstLine, secondLine] + shiftedLines, n + u, p + 2, n + u // 2

		# Only right child
		if self.leftChild == None:
			lines, n, p, x = self.rightChild.__displayAux()
			s = str(self.key)
			u = len(s)
			firstLine = s + x * "_" + (n - x) * " "
			secondLine = (u + x) * " " + "\\" + (n - x - 1) * " "
			shiftedLines = [u * " " + line for line in lines]
			return [firstLine, secondLine] + shiftedLines, n + u, p + 2, u // 2

		# Two children
		left, n, p, x = self.leftChild.__displayAux()
		right, m, q, y = self.rightChild.__displayAux()
		s = str(self.key)
		u = len(s)
		firstLine = (x + 1) * " " + (n - x - 1) * "_" + s + y * "_" + (m - y) * " "
		secondLine = x * " " + "/" + (n - x - 1 + u + y) * " " + "\\" + (m - y - 1) * " "
		if p < q:
			left += [n * " "] * (q - p)
		elif p > q:
			right += [m * " "] * (p - q)
		lines = [firstLine, secondLine] + [a + u * " " + b for a, b in zip(left, right)]
		return lines, n + m + u, max(p, q) + 2, n + u // 2

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def makeRandomBinaryTree():
	file = open("names.txt", "r")
	names = file.readlines()
	file.close()

	names = [name.replace("\n","") for name in names]

	treeKeys = random.sample(names, 15)

	binaryTree = Tree(treeKeys.pop())

	for name in treeKeys:
		binaryTree.insert(name)

	return binaryTree

def makeBalancedBST(data, lo = 0, hi = None):
	data.sort()

	if hi == None:
		hi = len(data) - 1
	if lo > hi:
		return None

	mid = (lo + hi) // 2

	root = Tree(data[mid])
	root.leftChild = makeBalancedBST(data, lo, mid - 1)
	root.rightChild = makeBalancedBST(data, mid + 1, hi)
	return root

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

while True:
	binaryTree = makeRandomBinaryTree()
	while binaryTree.isBalanced():
		binaryTree = makeRandomBinaryTree()

	print("{:>21}:  {}".format("Tree", binaryTree.toTuple()))
	print("{:>21}:  {}".format("Tree height", binaryTree.getHeight()))
	print("{:>21}:  {}".format("Is Binary Search Tree", binaryTree.isBST()[0]))
	print("{:>21}:  {}".format("Is balanced", binaryTree.isBalanced()), "\n")

	binaryTree.display()

	binaryTree = makeBalancedBST(binaryTree.listData())

	print("\n{:-^50}\n".format(" After balancing "))
	print("{:>21}:  {}".format("Tree", binaryTree.toTuple()))
	print("{:>21}:  {}".format("Tree height", binaryTree.getHeight()))
	print("{:>21}:  {}".format("In-order traversal", binaryTree.inOrderTraversal()))
	print("{:>21}:  {}".format("Pre-order traversal", binaryTree.preOrderTraversal()))
	print("{:>21}:  {}".format("Post-order traversal", binaryTree.postOrderTraversal()), "\n")

	binaryTree.display()

	choice = input("\nEnter to continue or X to exit: ").upper()
	if len(choice) > 0 and choice[0] == 'X':
		break
	print()
