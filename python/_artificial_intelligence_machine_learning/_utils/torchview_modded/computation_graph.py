# mypy: ignore-errors
from __future__ import annotations

from typing import Union, Any, Callable
from collections import Counter
from contextlib import nullcontext

from graphviz import Digraph
from torch.nn.modules import Identity

from .computation_node import NodeContainer
from .computation_node import TensorNode, ModuleNode, FunctionNode
from .utils import updated_dict, assert_input_type

COMPUTATION_NODES = Union[TensorNode, ModuleNode, FunctionNode]
NODE_TYPES = {
	TensorNode: 'tensor',
	ModuleNode: 'module',
	FunctionNode: 'function'
}


class ComputationGraph:
	"""A class to represent a model's computational graph"""

	def __init__(self, visual_graph: Digraph, root_container: NodeContainer[TensorNode], depth: int | float = 3):
		"""
		Resets the running_node_id, id_dict when a new ComputationGraph is initialized.
		Otherwise, labels would depend on previous ComputationGraph runs
		"""

		self.visual_graph = visual_graph
		self.root_container = root_container
		self.depth = depth
		self.node_info = dict()
		self.edge_node_ids = []
		self.subgraph_ids = []
		self.subgraph_labels = []

		self.reset_graph_history()

	def reset_graph_history(self) -> None:
		"""
		Resets to id config to the setting of empty visual graph
		needed for getting reproducible/deterministic node name and
		graphviz graphs. This is especially important for output tests
		"""

		self.context_tracker = {'current_context': [], 'current_depth': 0}
		self.running_node_id: int = 0
		self.running_subgraph_id: int = 0
		self.id_dict: dict[str, int] = {}
		self.node_set: set[int] = set()
		self.edge_list: list[tuple[COMPUTATION_NODES, COMPUTATION_NODES]] = []

		# module node  to capture whole graph
		main_container_module = ModuleNode(Identity(), -1)
		main_container_module.is_container = False
		self.subgraph_dict: dict[str, int] = {main_container_module.node_id: 0}
		self.running_subgraph_id += 1

		# Add input nodes
		self.node_hierarchy = {main_container_module: list(self.root_container)}
		for root_node in self.root_container:
			root_node.context = self.node_hierarchy[main_container_module]

	def fill_visual_graph(self) -> None:
		"""Fills the graphviz graph with desired nodes and edges."""

		self.render_nodes()
		self.render_edges()

	def render_nodes(self) -> None:
		kwargs = {
			'cur_node': self.node_hierarchy,
			'subgraph': None
		}
		self.traverse_graph(self.collect_graph, **kwargs)

	def render_edges(self) -> None:
		"""
		Records all edges in self.edge_list to
		the graphviz graph using node ids from edge_list
		"""

		edge_counter: dict[tuple[int, int], int] = {}

		for source, dest in self.edge_list:
			edge_id = self.id_dict[source.node_id], self.id_dict[dest.node_id]
			edge_counter[edge_id] = edge_counter.get(edge_id, 0) + 1
			self.add_edge(edge_id, edge_counter[edge_id])

	def traverse_graph(self, action_fn: Callable[..., None], **kwargs: Any) -> None:
		cur_node = kwargs['cur_node']
		cur_subgraph = self.visual_graph \
			if kwargs['subgraph'] is None \
			else kwargs['subgraph']
		assert_input_type(
			'traverse_graph', (TensorNode, ModuleNode, FunctionNode, dict), cur_node
		)
		if isinstance(cur_node, (TensorNode, ModuleNode, FunctionNode)):
			if cur_node.depth <= self.depth:
				action_fn(**kwargs)
			return

		if isinstance(cur_node, dict):
			k, v = list(cur_node.items())[0]
			new_kwargs = updated_dict(kwargs, 'cur_node', k)
			if 0 <= k.depth <= self.depth:
				action_fn(**new_kwargs)

			# if it is container module, move directly to outputs
			if k.is_container:
				for g in k.output_nodes:
					new_kwargs = updated_dict(new_kwargs, 'cur_node', g)
					self.traverse_graph(action_fn, **new_kwargs)
				return

			display_nested = 1 <= k.depth < self.depth

			with (
				cur_subgraph.subgraph(name=f'cluster_{self.subgraph_dict[k.node_id]}')
				if display_nested else nullcontext()
			) as cur_cont:
				if display_nested:
					cur_cont.attr(label=k.name)
					new_kwargs = updated_dict(new_kwargs, 'subgraph', cur_cont)
					self.subgraph_ids.append(f'cluster_{self.subgraph_dict[k.node_id]}')
					self.subgraph_labels.append(k.name)
				for g in v:
					new_kwargs = updated_dict(new_kwargs, 'cur_node', g)
					self.traverse_graph(action_fn, **new_kwargs)
				if self.subgraph_labels:
					self.subgraph_ids.pop()
					self.subgraph_labels.pop()

	def collect_graph(self, **kwargs: Any) -> None:
		"""Adds edges and nodes with appropriate node name/id"""

		cur_node = kwargs['cur_node']
		# if tensor node is traced, dont repeat collecting
		# if node is isolated, dont record it
		is_isolated = cur_node.is_root() and cur_node.is_leaf()
		if id(cur_node) in self.node_set or is_isolated:
			return

		self.check_node(cur_node)
		is_cur_visible = self.is_node_visible(cur_node)
		# add node
		if is_cur_visible:
			subgraph = kwargs['subgraph']
			if isinstance(cur_node, (FunctionNode, ModuleNode)):
				self.add_node(cur_node, subgraph)
			if isinstance(cur_node, TensorNode):
				self.add_node(cur_node, subgraph)

		elif isinstance(cur_node, ModuleNode):
			# add subgraph
			if cur_node.node_id not in self.subgraph_dict:
				self.subgraph_dict[cur_node.node_id] = self.running_subgraph_id
				self.running_subgraph_id += 1

		# add edges only through
		# node -> TensorNode -> Node connection
		if not isinstance(cur_node, TensorNode):
			return

		# add edges
		# {cur_node -> dest_node} part
		source_node = self.get_source_node(cur_node)
		is_main_node_visible = self.is_node_visible(cur_node.main_node)
		is_source_node_visible = self.is_node_visible(source_node)
		if not cur_node.is_leaf():
			for children_node in cur_node.children:
				is_output_visible = self.is_node_visible(children_node)
				if is_output_visible:
					if is_main_node_visible:
						self.edge_list.append((cur_node, children_node))
					elif is_source_node_visible:
						self.edge_list.append((source_node, children_node))

		# {source_node -> cur_node} part
		# visible tensor and non-input tensor nodes
		if is_cur_visible and not cur_node.is_root():
			assert not isinstance(source_node, TensorNode) or source_node.is_root(), \
				'get_source_node function returned inconsistent Node, please report this'
			self.edge_list.append((source_node, cur_node))

	def is_node_visible(self, compute_node: COMPUTATION_NODES) -> bool:
		"""
		Returns True if node should be displayed on the visual
		graph. Otherwise False
		"""

		assert_input_type('is_node_visible', (TensorNode, ModuleNode, FunctionNode), compute_node)

		if compute_node.name == 'empty-pass':
			return False

		if isinstance(compute_node, (ModuleNode, FunctionNode)):
			is_visible = isinstance(compute_node, FunctionNode) \
				or compute_node.is_container \
				or compute_node.depth == self.depth

			return is_visible

		else:
			if compute_node.main_node.depth < 0 or compute_node.is_aux:
				return False

			is_visible = (compute_node.is_root() or compute_node.is_leaf()) and compute_node.depth == 0

			return is_visible

	def get_source_node(self, _tensor_node: TensorNode) -> COMPUTATION_NODES:
		tensor_node = _tensor_node.main_node if _tensor_node.is_aux else _tensor_node

		# non-output nodes eminating from input node
		if tensor_node.is_root():
			return tensor_node

		current_parent_h = tensor_node.parent_hierarchy

		sorted_depth = sorted(depth for depth in current_parent_h)
		source_node = next(iter(tensor_node.parents))
		depth = 0
		for depth in sorted_depth:
			source_node = current_parent_h[depth]
			if depth >= self.depth:
				break

		module_depth = depth - 1
		if (
			isinstance(current_parent_h[depth], FunctionNode)
			and module_depth in tensor_node.parent_hierarchy
		):
			if current_parent_h[module_depth].is_container:
				return current_parent_h[module_depth]

		# Even though this is recursive, not harmful for complexity
		# The reason: the (time) complexity ~ O(L^2) where L
		# is the length of CONTINUOUS path along which the same tensor is passed
		# without any operation on it. L is always small since we dont use
		# infinitely big network with infinitely big continuou pass of unchanged
		# tensor. This recursion is necessary e.g. for LDC model
		if source_node.name == 'empty-pass':
			empty_pass_parent = next(iter(source_node.parents))
			assert isinstance(empty_pass_parent, TensorNode), (
				f'{empty_pass_parent} is input of {source_node}'
				f'and must a be TensorNode'
			)
			return self.get_source_node(empty_pass_parent)
		return source_node

	def add_node(self, node: COMPUTATION_NODES, subgraph: Digraph | None = None) -> None:
		"""
		Adds node to the graphviz with correct id, label and colour
		settings. Updates state of running_node_id if node is not
		identified before.
		"""

		if node.node_id not in self.id_dict:
			self.id_dict[node.node_id] = self.running_node_id
			self.running_node_id += 1

		label, node_id, node_name, node_type, *shapes = self.get_node_label(node)

		if subgraph is None or node.name in ('Input tensor', 'Output tensor'):
			subgraph = self.visual_graph
		subgraph.node(name=f'{self.id_dict[node.node_id]}', label=label)
		self.node_set.add(id(node))

		self.node_info[node_id] = {
			'name': node_name,
			'type': node_type,
			'shapes': shapes,
			'subgraph_ids': self.subgraph_ids[:],
			'subgraph_labels': self.subgraph_labels[:]
		}

	def add_edge(self, edge_ids: tuple[int, int], edg_cnt: int) -> None:
		source_node_id, dest_node_id = edge_ids
		label = None if edg_cnt == 1 else f' x{edg_cnt}'
		self.visual_graph.edge(f'{source_node_id}', f'{dest_node_id}', label=label)
		self.edge_node_ids.append((source_node_id, dest_node_id))

	def get_node_label(self, node: COMPUTATION_NODES) -> str:
		node_type = NODE_TYPES[type(node)]

		if isinstance(node, TensorNode):
			shape_repr = str(node.tensor_shape).replace('(1,', '(N,')  # Replace with N for batch size
			label = f'{node.name} {shape_repr}'

			return label, self.id_dict[node.node_id], node.name, node_type, shape_repr
		else:
			input_shape_repr = compact_list_repr(node.input_shape).replace('(1,', '(N,')
			output_shape_repr = compact_list_repr(node.output_shape).replace('(1,', '(N,')
			label = f'{node.name} {input_shape_repr} {output_shape_repr}'

			return label, self.id_dict[node.node_id], node.name, node_type, input_shape_repr, output_shape_repr

	def check_node(self, node: COMPUTATION_NODES) -> None:
		assert node.node_id != 'null', f'wrong id {node} {type(node)}'
		assert '-' not in node.node_id, 'No repetition of node recording is allowed'
		assert node.depth <= self.depth, f'Exceeds display depth limit, {node}'
		assert len(node.parents) in [0, 1] or not isinstance(node, TensorNode), \
			f'tensor must have single input node {node}'


def compact_list_repr(x: list[Any]) -> str:
	"""
	Returns more compact representation of list with
	repeated elements. This is useful for e.g. output of transformer/rnn
	models where hidden state outputs shapes is repetation of one hidden unit
	output
	"""

	list_counter = Counter(x)
	x_repr = ''

	for elem, cnt in list_counter.items():
		if cnt == 1:
			x_repr += f'{elem}, '
		else:
			x_repr += f'{cnt} x {elem}, '

	# Get rid of last comma
	return x_repr[:-2]


def get_output_id(dest_node: COMPUTATION_NODES) -> str:
	"""
	This returns id of output to get correct id.
	This is used to identify the recursively used modules.
	Identification relation is as follows:
		ModuleNodes => by id of nn.Module object
		Parameterless ModulesNodes => by id of nn.Module object
		FunctionNodes => by id of Node object
	"""

	if isinstance(dest_node, ModuleNode):
		output_id = str(dest_node.compute_unit_id)
	else:
		output_id = dest_node.node_id

	return output_id
