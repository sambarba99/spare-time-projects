"""
Decision tree plotter using GraphViz

Author: Sam Barba
Created 19/10/2022
"""

from graphviz import Digraph


def plot_tree(tree, features, labels):
	# def print_tree(tree, indent=0):
	# 	if tree.is_leaf:
	# 		print(f'{" " * indent}{labels[tree.class_idx]}')
	# 	else:
	# 		f = tree.feature_idx
	# 		print(f'{" " * indent}{features[f]} <= {tree.split_threshold}')
	# 		print_tree(tree.left, indent + 4)
	# 		print(f'{" " * indent}{features[f]} > {tree.split_threshold}')
	# 		print_tree(tree.right, indent + 4)

	def get_level(tree, level, nodes=None):
		"""Get the nodes of a certain tree level"""

		if nodes is None: nodes = []
		if not tree: return nodes

		if level == 0:
			s = labels[tree.class_idx] \
				if tree.is_leaf \
				else f'{features[tree.feature_idx]} <= {round(tree.split_threshold, 2)}'
			nodes.append(s)
		else:
			get_level(tree.left, level - 1, nodes)
			get_level(tree.right, level - 1, nodes)

		return nodes


	levels = []
	n = 0
	for i in range(tree.depth + 1):
		level = get_level(tree, i)
		for idx, node in enumerate(level):
			# GraphViz requires unique labels, so do this
			# E.g. a tree may have multiple 'setosa' leaf nodes
			# or multiple 'petal width <= 1.5' nodes
			level[idx] = f'{n}--{node}'
			n += 1
		levels.append(level)

	# 1. Set up global attributes

	g = Digraph(
		name='decision tree',
		node_attr={'fontname': 'consolas', 'fontsize': '11', 'style': 'filled,setlinewidth(0)', 'shape': 'rect'},
		edge_attr={'fontname': 'consolas', 'fontsize': '11', 'arrowsize': '0.7'}
	)

	# 2. Create nodes

	for level in levels:
		for node in level:
			# Colour feature split nodes blue, value (leaf) nodes green
			colour = '#80c0ff' if '<=' in node else '#30e090'
			g.node(node, label=node.split('--')[-1], color=colour)

	# 3. Create edges

	if len(levels) > 1:
		for level, next_level in zip(levels[:-1], levels[1:]):
			next_copy = next_level[:]
			for src in level:
				if '<=' not in src:
					continue  # Leaf nodes don't have children

				left_child = next_copy.pop(0)
				right_child = next_copy.pop(0)
				g.edge(src, left_child, label='T')  # Left = true
				g.edge(src, right_child, label='F')

	# 4. Render graph

	g.render('tree', view=True, cleanup=True, format='png')
