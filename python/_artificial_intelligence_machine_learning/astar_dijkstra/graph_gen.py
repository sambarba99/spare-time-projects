"""
Graph generator (as opposed to daedalus.py, which makes mazes)

Author: Sam Barba
Created 23/05/2022
"""

import numpy as np

from graph_node import GraphNode


def make_graph(x_max, y_max, num_nodes=1000, num_neighbours=4):
	"""Note: if max_neighbours = 1, a minimum spanning tree is created"""

	def dfs(visited, node):
		"""Depth-first search"""

		if node not in visited:
			visited.append(node)
			for neighbour in node.neighbours:
				dfs(visited, neighbour)


	x = np.random.uniform(10, x_max - 10, size=num_nodes)
	y = np.random.uniform(10, y_max - 10, size=num_nodes)
	coords = np.vstack((x, y)).T
	adjacency_list = dict()

	# Link each node to its nearest 'num_neighbours' neighbours
	for idx, c in enumerate(coords):
		if idx not in adjacency_list:
			adjacency_list[idx] = set()

		neighbour_dists = np.linalg.norm(coords - c, axis=1)
		# '1:' to exlude from nearest
		nearest_n = neighbour_dists.argsort()[1:num_neighbours + 1]

		for neighbour_idx in nearest_n:
			if neighbour_idx not in adjacency_list:
				adjacency_list[neighbour_idx] = set()
			adjacency_list[idx].add(neighbour_idx)
			adjacency_list[neighbour_idx].add(idx)

	# Create graph, as a list of node
	nodes = [GraphNode(idx, y, x) for idx, (x, y) in enumerate(coords)]
	for n in nodes:
		nodes[n.idx].neighbours = [nodes[i] for i in adjacency_list[n.idx]]

	# Check if graph is connected, via depth-first search (start arbitrarily from node 0)
	visited = []
	dfs(visited, nodes[0])
	connected = len(visited) == len(nodes)

	# Ensure graph connectivity
	while not connected:
		# Choose nearest pair of nodes in 'visited' and 'unvisited', and link them
		unvisited = [n for n in nodes if n not in visited]
		u_idx = v_idx = 0
		min_dist = 1e9
		for u in unvisited:
			for n in visited:
				if u is n: continue
				d = n.dist(u)
				if d < min_dist:
					min_dist = d
					u_idx, v_idx = u.idx, n.idx

		nodes[u_idx].neighbours.append(nodes[v_idx])
		nodes[v_idx].neighbours.append(nodes[u_idx])
		visited = []
		dfs(visited, nodes[0])
		connected = len(visited) == len(nodes)

	return nodes
