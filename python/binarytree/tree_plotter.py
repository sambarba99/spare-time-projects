"""
Binary tree plotter using GraphViz

Author: Sam Barba
Created 19/10/2022
"""

from graphviz import Graph

def plot_tree(tree, title):
	# 1. Set up global attributes

	g = Graph(name='binary tree',
		graph_attr={'splines': 'line'},
		node_attr={'style': 'filled,setlinewidth(0)', 'label': '', 'shape': 'rect', 'fillcolor': '#80c0ff'})

	# 2. Create nodes and edges

	nodes, edges = generate_graph(tree)

	for node in nodes:
		g.node(node, label=node)

	for n1, n2 in edges:
		g.edge(n1, n2)

	# 3. Render graph

	g.format = 'png'
	g.render(f'binary_tree_{title}', view=True, cleanup=True)

def generate_graph(tree, nodes=None, edges=None):
	if nodes is None: nodes = []
	if edges is None: edges = []

	if not tree: return nodes, edges

	nodes.append(tree.data)

	if tree.left_child:
		edges.append((tree.data, tree.left_child.data))
	if tree.right_child:
		edges.append((tree.data, tree.right_child.data))

	generate_graph(tree.left_child, nodes, edges)
	generate_graph(tree.right_child, nodes, edges)

	return nodes, edges
