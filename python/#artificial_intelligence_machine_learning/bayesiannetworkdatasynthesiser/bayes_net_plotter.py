"""
Bayesian network plotter using GraphViz

Author: Sam Barba
Created 02/12/2022
"""

from graphviz import Digraph
import numpy as np

def plot_bayes_net(adj_mat):
	# 1. Set up global attributes and key

	g = Digraph(name='bayesian network',
		graph_attr={'fontname': 'consolas', 'labelloc': 't', 'label': 'Bayesian Network'},
		node_attr={'style': 'filled,setlinewidth(0)', 'fontname': 'consolas', 'shape': 'circle'})

	g.node('key',
		label='<<table border="0" cellborder="1" cellspacing="0" cellpadding="5">'
			'<tr><td bgcolor="#30e090">Parent</td></tr>'
			'<tr><td bgcolor="#80c0ff">Child</td></tr></table>>',
		shape='rect',
		fillcolor='white')

	# 2. Create nodes

	for i in range(adj_mat.shape[0]):
		if adj_mat[:, i].sum() == 0:  # Parent node
			g.node(str(i), label=str(i), color='#30e090')
		else:  # Child node
			g.node(str(i), label=str(i), color='#80c0ff')

	# 3. Create edges

	for i in range(adj_mat.shape[0]):
		children = np.where(adj_mat[i] == 1)[0]
		for c in children:
			g.edge(str(i), str(c))

	g.format = 'png'
	g.render('bayes_net', view=True, cleanup=True)
