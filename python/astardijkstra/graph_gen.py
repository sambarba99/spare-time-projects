"""
Graph generator (as opposed to daedalus.py, which makes mazes)

Author: Sam Barba
Created 23/05/2022
"""

import numpy as np
from vertex_graph import GraphVertex

class GraphGen:
	"""Note: if max_neighbours = 1, a minimum spanning tree is created"""

	def __init__(self, max_x, max_y, n_vertices=500, max_neighbours=5):
		self.max_x = max_x
		self.max_y = max_y
		self.n_vertices = n_vertices
		self.max_neighbours = max_neighbours

	def make_graph(self):
		def dfs(visited, vertex):
			"""Depth-first search"""
			if vertex not in visited:
				visited.append(vertex)
				for neighbour in vertex.neighbours:
					dfs(visited, neighbour)

		x = np.random.uniform(10, self.max_x - 10, size=self.n_vertices)
		y = np.random.uniform(10, self.max_y - 10, size=self.n_vertices)
		coords = np.vstack((x, y)).T
		adjacency_list = dict()

		# Link each vertex to its closest N neighbours,
		# where N is random between 1 and max_neighbours
		for idx, c in enumerate(coords):
			if idx not in adjacency_list:
				adjacency_list[idx] = set()

			neighbour_dists = np.linalg.norm(coords - c, axis=1)
			# '1:' to exlude self from closest
			closest_n = np.argsort(neighbour_dists)[1:np.random.randint(1, self.max_neighbours + 1) + 1]

			for neighbour_idx in closest_n:
				if neighbour_idx not in adjacency_list:
					adjacency_list[neighbour_idx] = set()
				adjacency_list[idx].add(neighbour_idx)
				adjacency_list[neighbour_idx].add(idx)

		# Create graph, as a list of vertices
		vertices = [GraphVertex(idx, x, y) for idx, (x, y) in enumerate(coords)]
		for idx, v in enumerate(vertices):
			vertices[idx].neighbours = [vertices[i] for i in adjacency_list[v.idx]]

		# Check if graph is connected, via depth-first search (start arbitrarily from vertex no. 0)
		visited = []
		dfs(visited, vertices[0])
		connected = len(visited) == len(vertices)

		# Ensure graph connectivity
		while not connected:
			# Choose closest pair of vertices in 'visited' and 'unvisited', and link them
			unvisited = [v for v in vertices if v not in visited]
			u_idx = v_idx = 0
			min_dist = 1e9
			for u in unvisited:
				for v in visited:
					if u is v: continue
					d = v.dist(u)
					if d < min_dist:
						min_dist = d
						u_idx, v_idx = u.idx, v.idx

			vertices[u_idx].neighbours.append(vertices[v_idx])
			vertices[v_idx].neighbours.append(vertices[u_idx])
			visited = []
			dfs(visited, vertices[0])
			connected = len(visited) == len(vertices)

		return vertices
