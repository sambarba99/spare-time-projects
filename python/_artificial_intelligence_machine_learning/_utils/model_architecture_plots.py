"""
PyTorch model architecture plotting functionality

Author: Sam Barba
Created 22/03/2024
"""

from graphviz import Digraph
import torch
from torchsummary import summary


def make_node_label(layer_name, layer_type, input_shape, output_shape, activation=''):
	elements = [
		'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">',
		'<TR>',
		f'<TD COLSPAN="{bool(activation) + 1}" BGCOLOR="#7495a6:#aadaf2" GRADIENTANGLE="90">{layer_name}</TD>',
		'<TD BGCOLOR="#a6a6a6:#f2f2f2" GRADIENTANGLE="90">Input:</TD>',
		f'<TD BGCOLOR="#a6a6a6:#f2f2f2" GRADIENTANGLE="90">{input_shape}</TD>',
		'</TR>',
		'<TR>',
		f'<TD BGCOLOR="#4263a6:#6191f2" GRADIENTANGLE="90">{layer_type}</TD>',
		f'<TD BGCOLOR="#42a642:#61f261" GRADIENTANGLE="90">{activation}</TD>' if activation else '',
		'<TD BGCOLOR="#a6a6a6:#f2f2f2" GRADIENTANGLE="90">Output:</TD>',
		f'<TD BGCOLOR="#a6a6a6:#f2f2f2" GRADIENTANGLE="90">{output_shape}</TD>',
		'</TR>',
		'</TABLE>>'
	]
	return ''.join(elements)


def plot_model(model, input_shape, out_file='./model_architecture'):
	assert issubclass(model.__class__, torch.nn.Module)
	assert isinstance(input_shape, tuple)

	input_shape = tuple([1] + list(input_shape))  # Add batch size of 1

	# Get graph edges using symbolic_trace

	symbolic_trace = str(torch.fx.symbolic_trace(model).graph)
	symbolic_trace = symbolic_trace.replace('input_1', 'x').replace(' ', '')
	symbolic_trace = symbolic_trace.splitlines()[2:-1]
	symbolic_trace = [line for line in symbolic_trace if 'squeeze' not in line.lower()]  # Ignore shape operations
	src_nodes = [line.split('args=(%')[1].split(',)')[0].removeprefix('_') for line in symbolic_trace]
	src_nodes = [f'sequential_{i}' if len(i) == 1 else i for i in src_nodes]
	dest_nodes = [line.split('target=')[1].split(']')[0].replace('.', '_') for line in symbolic_trace]
	dest_nodes = [f'sequential_{i}' if len(i) == 1 else i for i in dest_nodes]

	edges = dict()
	for src_node, dest_node in zip(src_nodes, dest_nodes):
		if src_node in edges:
			edges[src_node].append(dest_node)
		else:
			edges[src_node] = [dest_node]

	# Get layer output shapes

	summary_str = str(summary(model, input_data=torch.zeros(input_shape), verbose=0))
	summary_str = summary_str.replace(' ', '').splitlines()[3:-11]
	summary_str = [line for line in summary_str if 'sequential' not in line.lower()]
	output_shapes = [line.split('[-1,')[1].split(']')[0] for line in summary_str]
	output_shapes = [f'(N,{i})'.replace(',', ', ') for i in output_shapes]

	# Get layer names, types, other info

	graph = dict()

	for name, layer in model.named_modules():
		if (not name) or isinstance(layer, torch.nn.modules.container.Sequential):
			continue
		name = name.replace('.', '_')
		name = f'sequential_{name}' if len(name) == 1 else name
		graph[name] = {
			'layer_type': layer.__class__.__name__,
			'is_activation': 'activation' in str(type(layer)),
			'input_shape': str(input_shape).replace('(1', '(N') if len(graph) == 0 else '',
			'output_shape': output_shapes[len(graph)],
			'activation': '',
			'src': 'x',
			'dest': edges.get(name, [])
		}

	# For every node, find its source node and input shape

	for k, v in graph.items():
		for vi in v['dest']:
			graph[vi]['src'] = k
			graph[vi]['input_shape'] = v['output_shape']

	# Merge any activation node into its source node

	merged_graph = graph.copy()

	for k, v in graph.items():
		if v['is_activation']:
			merged_graph[v['src']]['activation'] = v['layer_type']
			merged_graph[v['src']]['dest'] = v['dest']
			merged_graph.pop(k)  # Discard activation (now merged)

	# Create and render digraph

	g = Digraph(
		edge_attr={'arrowsize': '0.7', 'color': 'white'},
		graph_attr={'nodesep': '0.4', 'ranksep': '0.3', 'bgcolor': '#0d1117'},
		node_attr={'fontname': 'arial', 'fontsize': '10.5', 'shape': 'plain'}
	)

	for k, v in merged_graph.items():
		# Remove trailing underscore and digits from layer names e.g. conv_block_10 -> conv_block
		layer_name = k[:k.rfind('_')] if k.split('_')[-1].isdecimal() else k

		g.node(k,
			label=make_node_label(
				layer_name=layer_name,
				layer_type=v['layer_type'],
				input_shape=v['input_shape'],
				output_shape=v['output_shape'],
				activation=v['activation']
			)
		)

		for vi in v['dest']:
			g.edge(k, vi)

	g.render(out_file, view=True, cleanup=True, format='png')


def plot_model_manual(*, nodes, edges=None, out_file='./model_architecture'):
	def format_shape(shape_tuple):
		shape_tuple = tuple([1] + list(shape_tuple))
		shape_repr = str(shape_tuple).replace('(1', '(N')
		return shape_repr


	if edges is None:
		# Sequential model
		edges = [(i, i + 1) for i in range(len(nodes) - 1)]

	g = Digraph(
		edge_attr={'arrowsize': '0.7', 'color': 'white'},
		graph_attr={'nodesep': '0.4', 'ranksep': '0.3', 'bgcolor': '#0d1117'},
		node_attr={'fontname': 'arial', 'fontsize': '10.5', 'shape': 'plain'}
	)

	nodes[0]['input_shape'] = format_shape(nodes[0]['input_shape'])
	for node in nodes:
		node['output_shape'] = format_shape(node['output_shape'])
	for src, dest in edges:
		nodes[dest]['input_shape'] = nodes[src]['output_shape']

	for idx, node in enumerate(nodes):
		g.node(
			str(idx),
			label=make_node_label(
				layer_name=node['layer_name'],
				layer_type=node['layer_type'],
				input_shape=node['input_shape'],
				output_shape=node['output_shape'],
				activation=node.get('activation', '')
			)
		)

	for src, dest in edges:
		g.edge(str(src), str(dest))

	g.render(out_file, view=True, cleanup=True, format='png')
