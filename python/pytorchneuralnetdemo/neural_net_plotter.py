"""
Neural network plotter using GraphViz

Author: Sam Barba
Created 10/10/2022
"""

from graphviz import Digraph
from torch import nn

MAX_LAYER_NODES = 8

def plot_model(model):
	# Store layer names and sizes
	layers = [
		m for m in model.modules()
		if hasattr(m, 'in_features') or isinstance(m, nn.Dropout)
	]
	layers.insert(0, layers[0])
	layer_names_sizes = [(None, None)] * len(layers)
	for idx, l in enumerate(layers):
		if idx == 0:
			layer_names_sizes[idx] = (None, l.in_features)
		else:
			if isinstance(l, nn.Dropout):
				layer_names_sizes[idx] = (l.__class__.__name__, layers[idx - 1].out_features)
			else:
				layer_names_sizes[idx] = (l.__class__.__name__, l.out_features)

	input_layer = layer_names_sizes[0]
	hidden_layers = layer_names_sizes[1:-1]
	output_layer = layer_names_sizes[-1]

	# 1. Set up global attributes

	g = Digraph(name='neural net',
		graph_attr={'rankdir': 'LR', 'splines': 'line', 'nodesep': '0.05'},
		node_attr={'style': 'filled,setlinewidth(0)', 'label': '', 'shape': 'circle'},
		edge_attr={'penwidth': '0.5', 'arrowsize': '0.5'})

	# 2. Create nodes

	# Input layer
	with g.subgraph(name='cluster_input') as c:
		_, size = input_layer
		size_limit = min(MAX_LAYER_NODES, size)
		if size > size_limit:
			c.attr(label=f'Input layer (+ {size - size_limit})')
		else:
			c.attr(label=f'Input layer')
		c.attr(color='white')
		c.node_attr.update(fillcolor='#80c0ff')
		for i in range(size_limit):
			c.node(f'in_{i}')

	# Hidden layers
	for layer_idx, layer in enumerate(hidden_layers):
		with g.subgraph(name=f'cluster_hidden_{layer_idx + 1}') as c:
			name, size = layer
			size_limit = min(MAX_LAYER_NODES, size)
			if size > size_limit:
				c.attr(label=f'Hidden layer {layer_idx + 1} ({name}) (+ {size - size_limit})')
			else:
				c.attr(label=f'Hidden layer {layer_idx + 1} ({name})')
			c.attr(color='white')
			c.node_attr.update(fillcolor='#c090ff')
			for i in range(size_limit):
				c.node(f'hid_{layer_idx}_{i}')

	# Output layer
	with g.subgraph(name='cluster_output') as c:
		name, size = output_layer
		size_limit = min(MAX_LAYER_NODES, size)
		if size > size_limit:
			c.attr(label=f'Output layer ({name}) (+ {size - size_limit})')
		else:
			c.attr(label=f'Output layer ({name})')
		c.attr(color='white')
		c.node_attr.update(fillcolor='#30e090')
		for i in range(size_limit):
			c.node(f'out_{i}')

	# 3. Create edges

	# Connect input layer to 1st hidden layer
	for i in range(min(MAX_LAYER_NODES, input_layer[1])):
		for j in range(min(MAX_LAYER_NODES, hidden_layers[0][1])):
			g.edge(f'in_{i}', f'hid_0_{j}')

	# Connect hidden layers
	if len(hidden_layers) > 1:
		for layer_idx in range(len(hidden_layers) - 1):
			this_layer_size = min(MAX_LAYER_NODES, hidden_layers[layer_idx][1])
			next_layer_size = min(MAX_LAYER_NODES, hidden_layers[layer_idx + 1][1])
			for i in range(this_layer_size):
				for j in range(next_layer_size):
					g.edge(f'hid_{layer_idx}_{i}', f'hid_{layer_idx + 1}_{j}')

	# Connect last hidden layer to output layer
	for i in range(min(MAX_LAYER_NODES, hidden_layers[-1][1])):
		for j in range(min(MAX_LAYER_NODES, output_layer[1])):
			g.edge(f'hid_{len(hidden_layers) - 1}_{i}', f'out_{j}')

	# 4. Render graph

	g.format = 'png'
	g.render('neural_net', view=True, cleanup=True)

	# with open('neural_net_graph_src.gz', 'w') as file:
	# 	file.write(g.source)
