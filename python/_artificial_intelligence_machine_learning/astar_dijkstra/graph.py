"""
Graph class for A* and Dijkstra demo

Author: Sam Barba
Created 05/10/2024
"""

import heapq
import math
import random

import numpy as np

from node import Node


class Graph:
	def __init__(self, graph_type, rows=None, cols=None, num_nodes=None, max_edges_per_node=None, x_max=None, y_max=None):
		assert graph_type in ('labyrinth', 'maze', 'graph')

		if graph_type in ('labyrinth', 'maze'):
			assert None not in (rows, cols)
			assert (rows % 2) and (cols % 2), 'Rows and cols must be odd'
		else:
			assert None not in (num_nodes, max_edges_per_node, x_max, y_max)

		self.graph_type = graph_type
		self.rows = rows
		self.cols = cols
		self.max_edges_per_node = max_edges_per_node
		self.x_max = x_max
		self.y_max = y_max

		self.nodes = []
		self.make_graph(num_nodes)

		if self.graph_type in ('labyrinth', 'maze'):
			# Start and target are top-left and bottom-left, respectively
			self.start_node = self[0, 0]
			self.target_node = self[self.rows - 1, self.cols - 1]
		else:
			# Start is top-left-most node (nearest to 0,0); target is node furthest from this
			node_coords = np.array([[node.x, node.y] for node in self])
			distances_from_top_left = np.linalg.norm(node_coords, axis=1)
			self.start_node = self[distances_from_top_left.argmin()]
			self.target_node = self[distances_from_top_left.argmax()]

	def make_graph(self, num_nodes):
		def remove_walls(a, b):
			mid_y = (a.y + b.y) // 2
			mid_x = (a.x + b.x) // 2
			self[mid_y, mid_x].is_wall = False
			self[a.y, a.x].is_wall = False
			self[b.y, b.x].is_wall = False

		def dfs(start_node):
			"""Non-recursive depth-first search"""

			open_set = [start_node]
			closed_set = set()

			while open_set:
				node = open_set.pop()
				closed_set.add(node)

				# Add all unvisited neighbours to the stack
				neighbours = self.get_neighbours(node)
				unvisited = neighbours.difference(closed_set)
				open_set.extend(unvisited)

			return closed_set


		if self.graph_type in ('labyrinth', 'maze'):
			self.nodes = [Node(x, y) for y in range(self.rows) for x in range(self.cols)]

			# Start at top-left
			self[0, 0].is_wall = False

			if self.graph_type == 'labyrinth':
				# Remove walls to carve a path

				nodes_to_visit = [self[0, 0]]
				visited = {self[0, 0]}

				while nodes_to_visit:
					current = nodes_to_visit[-1]  # Peek the top of the stack
					walls = self.get_surrounding_walls(current)
					unvisited_walls = walls.difference(visited)

					if unvisited_walls:
						next_wall = random.choice(list(unvisited_walls))
						remove_walls(current, next_wall)
						nodes_to_visit.append(next_wall)
						visited.add(next_wall)
					else:
						nodes_to_visit.pop()  # Backtrack if no unvisited walls are found
			else:
				# Remove 2/3 of the walls randomly

				wall_coords = [(node.y, node.x) for node in self if node.is_wall]
				num_walls_to_remove = int(len(wall_coords) * 2 / 3)

				# Loop until a valid graph is generated
				while True:
					# Reset neighbours and is_wall
					self[0, 0].neighbours = set()
					for y, x in wall_coords:
						self[y, x].is_wall = True
						self[y, x].neighbours = set()

					random.shuffle(wall_coords)
					for i in range(num_walls_to_remove):
						y, x = wall_coords[i]
						self[y, x].is_wall = False

					# End at bottom-right
					self[self.rows - 1, self.cols - 1].is_wall = False

					# Check target (bottom-right) can be reached from start (top-left), via depth-first search
					visited = dfs(self[0, 0])
					if self[self.rows - 1, self.cols - 1] in visited:
						break

				# Make any remaining non-traversible gaps into walls
				unvisited = set(self).difference(visited)
				for node in unvisited:
					node.is_wall = True

		else:  # Graph (nodes/edges)
			# Ensure random points aren't too close

			min_dist = 15
			while len(self) < num_nodes:
				x, y = random.uniform(20, self.x_max - 20), random.uniform(20, self.y_max - 20)
				distance_valid = True
				for node in self:
					if math.dist((x, y), (node.x, node.y)) < min_dist:
						distance_valid = False
						break
				if distance_valid:
					self.nodes.append(Node(x, y, len(self)))

			# Make graph into a Minimum Spanning Tree to ensure connectivity

			# Priority queue (distance, node_in, node_out)
			priority_queue = []
			in_tree = [self[0]]  # Start arbitrarily from node 0
			out_tree = self[1:]

			# Initialise priority queue with all distances from start
			for node_out in out_tree:
				dist = self.dist(self[0], node_out)
				heapq.heappush(priority_queue, (dist, self[0], node_out))

			while out_tree:
				# Get the pair of nodes in the tree and out of the tree that are nearest
				_, nearest_in, nearest_out = heapq.heappop(priority_queue)

				if nearest_out in in_tree \
					or len(nearest_in.neighbours) >= self.max_edges_per_node \
					or len(nearest_out.neighbours) >= self.max_edges_per_node:
					continue

				# Connect both, and add nearest_out to in_tree
				nearest_in.neighbours.add(nearest_out)
				nearest_out.neighbours.add(nearest_in)
				in_tree.append(nearest_out)
				out_tree.remove(nearest_out)

				# Update priority queue with distances for the new node added to in_tree
				for node_out in out_tree:
					dist = self.dist(nearest_out, node_out)
					heapq.heappush(priority_queue, (dist, nearest_out, node_out))

			# Add some more random edges so that the graph isn't just an MST

			node_coords = np.array([[node.x, node.y] for node in self])

			for idx, c in enumerate(node_coords):
				neighbour_dists = np.linalg.norm(node_coords - c, axis=1)
				nearest_n = neighbour_dists.argsort()[1:]  # '1:' to exlude self from nearest
				edges_to_add = random.randint(1, self.max_edges_per_node - 1)  # -1 because of previous MST step
				done_edges = 0
				for neighbour_idx in nearest_n:
					if len(self[idx].neighbours) >= self.max_edges_per_node or done_edges == edges_to_add:
						break
					if len(self[neighbour_idx].neighbours) >= self.max_edges_per_node:
						continue
					self[idx].neighbours.add(self[neighbour_idx])
					self[neighbour_idx].neighbours.add(self[idx])
					done_edges += 1

	def __len__(self):
		return len(self.nodes)

	def __getitem__(self, index):
		try:
			# Getting index from a row and col
			y, x = index

			if y in range(self.rows) and x in range(self.cols):
				return self.nodes[self.cols * y + x]

			return None
		except:
			# Using an explicit index
			return self.nodes[index]

	def reset_node_parents(self):
		for node in self:
			node.parent = None

	def dist(self, node_a, node_b):
		if self.graph_type in ('labyrinth', 'maze'):
			# Manhattan distance
			return abs(node_a.x - node_b.x) + abs(node_a.y - node_b.y)

		# Euclidean
		return math.dist((node_a.x, node_a.y), (node_b.x, node_b.y))

	def get_neighbours(self, node):
		if node.neighbours:
			return node.neighbours

		neighbours = set()

		for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # N, E, S, W
			neighbour = self[node.y + dy, node.x + dx]
			if neighbour and not neighbour.is_wall:
				neighbours.add(neighbour)

		node.neighbours = neighbours

		return neighbours

	def get_surrounding_walls(self, node):
		"""
		For labyrinth generation, get the set of surrounding walls at a distance of 2.
		At each generation step, a given wall 'a' and a randomly selected neighbour wall 'b'
		are removed, including the wall in between. The result is a 'carved out' labyrinth.
		"""

		neighbours = set()

		for dy, dx in [(-2, 0), (0, 2), (2, 0), (0, -2)]:  # N, E, S, W
			neighbour = self[node.y + dy, node.x + dx]
			if neighbour and neighbour.is_wall:
				neighbours.add(neighbour)

		return neighbours
