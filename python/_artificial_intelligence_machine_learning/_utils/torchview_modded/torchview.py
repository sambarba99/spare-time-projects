from contextlib import ExitStack
from typing import Any, Callable, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

from graphviz import Digraph
import torch
from torch import nn

from .computation_node import NodeContainer
from .computation_graph import ComputationGraph
from .computation_node import TensorNode
from .recorder_tensor import (
	_orig_module_forward, collect_tensor_node, module_forward_wrapper,
	Recorder, RecorderTensor, reduce_data_info
)


INPUT_DATA_TYPE = Union[torch.Tensor, Sequence[Any], Mapping[str, Any]]
CORRECTED_INPUT_DATA_TYPE = Optional[Union[Iterable[Any], Mapping[Any, Any]]]
INPUT_SIZE_TYPE = Sequence[Union[int, Sequence[Any], torch.Size]]
CORRECTED_INPUT_SIZE_TYPE = List[Union[Sequence[Any], torch.Size]]


def draw_graph(model, input_data=None, device=None, **kwargs):
	# Create computation graph as usual

	input_recorder_tensor, kwargs_record_tensor, input_nodes = process_input(
		input_data, kwargs, device
	)

	temp_digraph = Digraph()
	computation_graph = ComputationGraph(temp_digraph, input_nodes)

	# Get the names of sequential modules, in the order in which they're executed

	sequential_names_and_modules = [
		(name, module) for name, module in model.named_modules()
		if isinstance(module, nn.Sequential)
	]
	if sequential_names_and_modules:
		sequential_names, sequential_modules = zip(*sequential_names_and_modules)
		ordered_sequential_names = []
		hook_handles = []

		def hook_func(module, input, output):
			module_idx = sequential_modules.index(module)
			ordered_sequential_names.append(sequential_names[module_idx])

		for module in sequential_modules:
			hook_handle = module.register_forward_hook(hook_func)
			hook_handles.append(hook_handle)

		_ = model(*input_data)  # Populate ordered_sequential_names

		for h in hook_handles:
			h.remove()  # Remove hook after use

		# In case we get e.g. ['transformer.0.fc_block', 'transformer.1.fc_block', 'transformer']
		# we want to sort it to get ['transformer', 'transformer.0.fc_block', 'transformer.1.fc_block']
		# (i.e. for each subgroup starting with a certain prefix, put the prefix first)

		try:
			prefix_set = set(name.split('.')[0] for name in ordered_sequential_names)
			prefix_set = sorted(prefix_set, key=ordered_sequential_names.index)
			sorted_sub_groups = []
			for name_prefix in prefix_set:
				names_with_prefix = [name for name in ordered_sequential_names if name.startswith(name_prefix)]
				sorted_sub_groups.extend(sorted(names_with_prefix))

			ordered_sequential_names = sorted_sub_groups[:]
		except:
			# There is no node whose name is a prefix for other nodes, so can continue
			pass
	else:
		ordered_sequential_names = []

	# Fill computation visual graph (temp_digraph) as usual

	forward_prop(
		ordered_sequential_names, model, input_recorder_tensor, device, computation_graph, **kwargs_record_tensor
	)

	computation_graph.fill_visual_graph()

	# Modified visual graph section: start by defining new graph attributes

	model_digraph = Digraph(
		graph_attr={'ordering': 'in', 'rankdir': 'TD', 'nodesep': '0.4', 'ranksep': '0.3', 'bgcolor': '#0d1117'},
		node_attr={'shape': 'plain', 'style': 'filled', 'fontname': 'arial', 'fontsize': '10'},
		edge_attr={'arrowsize': '0.8', 'color': 'white', 'fontname': 'arial', 'fontsize': '10', 'fontcolor': 'white'}
	)

	# Merge activation nodes into their previous nodes (and reconnecting links to outputs)

	activation_node_ids_and_names = {
		node_id: node_info['name'] for node_id, node_info in computation_graph.node_info.items()
		if node_info['name'] in nn.modules.activation.__all__
	}
	for act_node_id, act_node_name in activation_node_ids_and_names.items():
		source_node_ids = [src for src, dest in computation_graph.edge_node_ids if dest == act_node_id]
		for source_node_id in source_node_ids:
			# Concatenate activation name to its parent node
			# E.g. If the activation node is ReLU and its source is Linear, its source name becomes 'Linear | ReLU'
			computation_graph.node_info[source_node_id]['name'] += f' | {act_node_name}'

	remaining_nodes = {
		node_id: node_info for node_id, node_info in computation_graph.node_info.items()
		if node_id not in activation_node_ids_and_names
	}
	remaining_edges = [
		(source, dest) for source, dest in computation_graph.edge_node_ids
		if source not in activation_node_ids_and_names
		and dest not in activation_node_ids_and_names
	]

	for source, dest in computation_graph.edge_node_ids:
		if source in activation_node_ids_and_names:
			# If the edge source is an activation, find its source(s), and link those to the dest
			# (i.e. skip over the activation node, as it was removed in remaining_nodes)
			sources_of_act_node = [source1 for source1, dest1 in computation_graph.edge_node_ids if dest1 == source]
			remaining_edges.extend((source1, dest) for source1 in sources_of_act_node)

	# Add nodes

	for node_id, node_info in remaining_nodes.items():
		if len(node_info['shapes']) == 1:
			label = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">
					<TR>
						<TD BGCOLOR="#44aa59:#66ff85" GRADIENTANGLE="90">{node_info['name']}</TD>
						<TD BGCOLOR="#aaaaaa:#ffffff" GRADIENTANGLE="90">{node_info['shapes'][0]}</TD>
					</TR>
				</TABLE>>'''
		else:  # 2 (input and output)
			has_activation = ' | ' in node_info['name']
			if has_activation:
				op_name, activation = node_info['name'].split(' | ')
				activation_row = f'<TD BGCOLOR="#ff8b00:#ffd000" GRADIENTANGLE="90">{activation}</TD>'
			else:
				op_name = node_info['name']
				activation_row = ''
			colour = '#4774aa:#6baeff' if node_info['type'] == 'module' else '#717171:#aaaaaa'
			label = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">
					<TR>
						<TD ROWSPAN="{1 if has_activation else 2}" BGCOLOR="{colour}" GRADIENTANGLE="90">{op_name}</TD>
						<TD BGCOLOR="#aaaaaa:#ffffff" GRADIENTANGLE="90">Input</TD>
						<TD BGCOLOR="#aaaaaa:#ffffff" GRADIENTANGLE="90">{node_info['shapes'][0]}</TD>
					</TR>
					<TR>
						{activation_row}
						<TD BGCOLOR="#aaaaaa:#ffffff" GRADIENTANGLE="90">Output</TD>
						<TD BGCOLOR="#aaaaaa:#ffffff" GRADIENTANGLE="90">{node_info['shapes'][1]}</TD>
					</TR>
				</TABLE>>'''

		if node_info['subgraph_ids'] and node_info['name'] not in ('Input tensor', 'Output tensor'):
			current_subgraph = model_digraph
			with ExitStack() as stack:
				for subgraph_id, subgraph_label in zip(node_info['subgraph_ids'], node_info['subgraph_labels']):
					subgraph = stack.enter_context(current_subgraph.subgraph(name=subgraph_id))
					subgraph.attr(
						label=subgraph_label, labeljust='l', color='white', style='dashed',
						fontname='arial', fontsize='10', fontcolor='white'
					)
					subgraph.node(str(node_id), label=label)
					current_subgraph = subgraph
		else:
			model_digraph.node(str(node_id), label=label)

	# Add edges

	for src, dest in remaining_edges:
		model_digraph.edge(str(src), str(dest))

	return model_digraph


def forward_prop(sequential_module_names, model, x, device, model_graph, **kwargs) -> None:
	"""
	Performs forward propagation of model on RecorderTensor
	inside context to use module_forward_wrapper
	"""

	saved_model_mode = model.training
	try:
		model.eval()
		new_module_forward = module_forward_wrapper(sequential_module_names, model_graph)
		with Recorder(_orig_module_forward, new_module_forward, model_graph):
			with torch.no_grad():
				if isinstance(x, (list, tuple)):
					_ = model.to(device)(*x, **kwargs)
				elif isinstance(x, Mapping):
					_ = model.to(device)(**x, **kwargs)
				else:
					raise ValueError('Unknown input type')
	except Exception as e:
		raise RuntimeError('Failed to run torchgraph see error message') from e
	finally:
		model.train(saved_model_mode)


def process_input(input_data, kwargs, device):
	"""Reads sample input data to get the input size."""

	x = None
	kwargs_recorder_tensor = traverse_data(kwargs, get_recorder_tensor, type)
	if input_data is not None:
		x = set_device(input_data, device)
		x = traverse_data(x, get_recorder_tensor, type)
		if isinstance(x, RecorderTensor):
			x = [x]

	input_data_node: NodeContainer[TensorNode] = (
		reduce_data_info(
			[x, kwargs_recorder_tensor], collect_tensor_node, NodeContainer()
		)
	)
	return x, kwargs_recorder_tensor, input_data_node


def traverse_data(data: Any, action_fn: Callable[..., Any], aggregate_fn: Callable[..., Any]) -> Any:
	"""
	Traverses any type of nested data. On a tensor, returns the action given by
	action_fn, and afterwards aggregates the results using aggregate_fn.
	"""

	if isinstance(data, torch.Tensor):
		return action_fn(data)

	# Recursively apply to collection items
	aggregate = aggregate_fn(data)
	if isinstance(data, Mapping):
		return aggregate(
			{
				k: traverse_data(v, action_fn, aggregate_fn)
				for k, v in data.items()
			}
		)
	if isinstance(data, tuple) and hasattr(data, '_fields'):  # Named tuple
		return aggregate(
			*(traverse_data(d, action_fn, aggregate_fn) for d in data)
		)
	if isinstance(data, Iterable) and not isinstance(data, str):
		return aggregate(
			[traverse_data(d, action_fn, aggregate_fn) for d in data]
		)
	# Data is neither a tensor nor a collection
	return data


def set_device(data: Any, device: torch.device | str) -> Any:
	"""Sets device for all data types and collections of input types."""

	return traverse_data(
		data,
		action_fn=lambda data: data.to(device, non_blocking=True),
		aggregate_fn=type,
	)


def get_recorder_tensor(input_tensor: torch.Tensor) -> RecorderTensor:
	"""
	Returns RecorderTensor version of input_tensor with
	TensorNode instance attached to it
	"""

	# as_subclass is necessary for torch versions < 3.12
	input_recorder_tensor: RecorderTensor = input_tensor.as_subclass(RecorderTensor)
	input_recorder_tensor.tensor_nodes = []
	input_node = TensorNode(tensor=input_recorder_tensor, depth=0, name='Input tensor')

	input_recorder_tensor.tensor_nodes.append(input_node)
	return input_recorder_tensor


def get_input_tensor(input_size, dtypes, device):
	"""Get input_tensor for use in model.forward()"""

	x = []
	for size, dtype in zip(input_size, dtypes):
		input_tensor = torch.rand(*size)
		x.append(
			get_recorder_tensor(input_tensor.to(device).type(dtype))
		)
	return x


def get_correct_input_sizes(input_size: INPUT_SIZE_TYPE) -> CORRECTED_INPUT_SIZE_TYPE:
	"""
	Convert input_size to the correct form, which is a list of tuples.
	Also handles multiple inputs to the network.
	"""

	if not isinstance(input_size, (list, tuple)):
		raise TypeError(
			'Input_size is not a recognized type. Please ensure input_size is valid.\n'
			'For multiple inputs to the network, ensure input_size is a list of tuple '
			'sizes. If you are having trouble here, please submit a GitHub issue.'
		)
	if not input_size or any(size <= 0 for size in flatten(input_size)):
		raise ValueError('Input_data is invalid, or negative size found in input_data.')

	if isinstance(input_size, list) and isinstance(input_size[0], int):
		return [tuple(input_size)]
	if isinstance(input_size, list):
		return input_size
	if isinstance(input_size, tuple) and isinstance(input_size[0], tuple):
		return list(input_size)
	return [input_size]


def flatten(nested_array: INPUT_SIZE_TYPE) -> Iterator[Any]:
	for item in nested_array:
		if isinstance(item, (list, tuple)):
			yield from flatten(item)
		else:
			yield item
