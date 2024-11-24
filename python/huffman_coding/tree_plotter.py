"""
Huffman tree plotter using GraphViz

Author: Sam Barba
Created 22/11/2022
"""

from graphviz import Graph


def plot_tree(tree):
	# Set up global attributes

	g = Graph(
		name='Huffman tree',
		graph_attr={'splines': 'line'},
		node_attr={'style': 'filled,setlinewidth(0)', 'shape': 'rect', 'fontname': 'consolas'},
		edge_attr={'fontname': 'consolas'}
	)

	# Create nodes and edges

	nodes, edges = generate_graph(tree)

	for node in nodes:
		symbol, weight = node.split('_')
		if len(symbol) == 1:  # Leaf node (just 1 char)
			g.node(node, label=f"'{symbol}' ({weight})", fillcolor='#30e090')
		else:
			g.node(node, label=weight, fillcolor='#80c0ff')

	for n1, n2, bit in edges:
		g.edge(n1, n2, label=str(bit))

	# Render graph

	g.render('tree', view=True, cleanup=True, format='png')


def generate_graph(tree, nodes=None, edges=None):
	def make_str(tree):
		return f'{tree.symbol}_{tree.weight:.2g}'


	if nodes is None: nodes = []
	if edges is None: edges = []

	if not tree: return nodes, edges

	nodes.append(make_str(tree))

	if tree.left_child:
		edges.append((make_str(tree), make_str(tree.left_child), 0))
	if tree.right_child:
		edges.append((make_str(tree), make_str(tree.right_child), 1))

	generate_graph(tree.left_child, nodes, edges)
	generate_graph(tree.right_child, nodes, edges)

	return nodes, edges
