"""
A* and Dijkstra demo

Author: Sam Barba
Created 20/09/2021
"""

import tkinter as tk

import pygame as pg

from graph import Graph


# For labyrinth/maze mode
ROWS = 59
COLS = 99
CELL_SIZE = 10

# For graph (nodes/edges) mode
NUM_NODES = 1000
MAX_EDGES_PER_NODE = 4
WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE

graph = None


def generate_graph():
	global graph

	graph = Graph(
		graph_type=stringv_type.get(),
		rows=ROWS, cols=COLS, num_nodes=NUM_NODES, max_edges_per_node=MAX_EDGES_PER_NODE,
		x_max=WIDTH, y_max=HEIGHT
	)

	draw(open_set=[], closed_set=set(), is_solved=False)


def astar():
	graph.reset_node_parents()

	open_set = [graph.start_node]
	closed_set = set()

	# Costs nothing to get from start to start (start_node parent will always be None)
	graph.start_node.g_cost = 0
	graph.start_node.h_cost = graph.dist(graph.start_node, graph.target_node)

	while open_set:
		cheapest_node = min(open_set, key=lambda node: (node.f_cost, node.h_cost))

		if cheapest_node is graph.target_node:
			break  # Found target

		draw(open_set, closed_set, False)
		open_set.remove(cheapest_node)
		closed_set.add(cheapest_node)

		neighbours = graph.get_neighbours(cheapest_node)
		unvisited = neighbours.difference(closed_set)
		for neighbour in unvisited:
			tentative_g_cost = cheapest_node.g_cost + graph.dist(cheapest_node, neighbour)
			if neighbour not in open_set:
				open_set.append(neighbour)  # Discovered a new node
			elif tentative_g_cost >= neighbour.g_cost:
				continue  # This isn't a better path

			# This path to neighbour is better than any previous one - record it
			neighbour.parent = cheapest_node
			neighbour.g_cost = tentative_g_cost
			neighbour.h_cost = graph.dist(neighbour, graph.target_node)

	draw(open_set, closed_set, True)


def dijkstra():
	"""Generates Shortest Path Tree"""

	graph.reset_node_parents()

	open_set = [graph.start_node]
	closed_set = set()

	# Costs nothing to get from start to start (start_node parent will always be None)
	graph.start_node.cost = 0

	while open_set:
		cheapest_node = min(open_set, key=lambda node: node.cost)

		if cheapest_node is graph.target_node:
			break  # Found target

		draw(open_set, closed_set, False)
		open_set.remove(cheapest_node)
		closed_set.add(cheapest_node)

		neighbours = graph.get_neighbours(cheapest_node)
		unvisited = neighbours.difference(closed_set)
		for neighbour in unvisited:
			tentative_cost = cheapest_node.cost + graph.dist(cheapest_node, neighbour)
			if neighbour not in open_set:
				open_set.append(neighbour)  # Discovered a new node
			elif tentative_cost >= neighbour.cost:
				continue  # This isn't a better path

			# This path to neighbour is better than any previous one - record it
			neighbour.parent = cheapest_node
			neighbour.cost = tentative_cost

	draw(open_set, closed_set, True)


def graph_traversal(search_type):
	"""Non-recursive traversal search (depth-first or breadth-first)"""

	assert search_type in ('dfs', 'bfs')

	graph.reset_node_parents()

	open_set = [graph.start_node]
	closed_set = set()

	while open_set:
		if graph.target_node.parent:
			break  # Found target

		draw(open_set, closed_set, False)

		pop_idx = -1 if search_type == 'dfs' else 0
		node = open_set.pop(pop_idx)
		closed_set.add(node)

		neighbours = graph.get_neighbours(node)
		unvisited = neighbours.difference(closed_set)
		for neighbour in unvisited:
			if neighbour not in open_set:
				open_set.append(neighbour)  # Discovered a new node
			neighbour.parent = node

	draw(open_set, closed_set, True)


def draw(open_set, closed_set, is_solved):
	scene.fill('black' if stringv_type.get() in ('labyrinth', 'maze') else (192, 192, 192))

	path = None

	if is_solved:
		# Trace back from end
		path = []
		current = graph.target_node
		while current:
			path.insert(0, current)
			current = current.parent

	if stringv_type.get() in ('labyrinth', 'maze'):
		for node in graph:
			if node in (graph.start_node, graph.target_node):
				colour = (255, 64, 0)
			elif node.is_wall:
				colour = (80, 80, 80)
			elif node in open_set:
				colour = (0, 192, 0)
			elif node in closed_set:
				colour = (64, 128, 192)
			else:
				colour = (192, 192, 192)

			pg.draw.rect(scene, colour, pg.Rect(node.x * CELL_SIZE, node.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
		pg.display.update()
		clock.tick(90)

		if not path:
			return

		for i in range(len(path) - 1):
			node = path[i]
			next_node = path[i + 1]
			pg.draw.line(
				scene, 'black',
				((node.x + 0.5) * CELL_SIZE - 1, (node.y + 0.5) * CELL_SIZE - 1),
				((next_node.x + 0.5) * CELL_SIZE - 1, (next_node.y + 0.5) * CELL_SIZE - 1),
				2
			)
			pg.draw.circle(
				scene,
				'black',
				((node.x + 0.5) * CELL_SIZE, (node.y + 0.5) * CELL_SIZE),
				1
			)
			pg.draw.circle(
				scene,
				'black',
				((next_node.x + 0.5) * CELL_SIZE, (next_node.y + 0.5) * CELL_SIZE),
				1
			)
			pg.display.update()
			clock.tick(120)

	else:
		# Draw edges then nodes on top
		done_edges = set()
		for node in graph:
			for neighbour in graph.get_neighbours(node):
				pair = tuple(sorted([node.idx, neighbour.idx]))
				if pair in done_edges:
					continue
				done_edges.add(pair)
				pg.draw.line(scene, 'black', (node.x - 1, node.y - 1), (neighbour.x - 1, neighbour.y - 1))

		if path:
			for i in range(len(path) - 1):
				node = path[i]
				next_node = path[i + 1]
				pg.draw.line(scene, 'black', (node.x, node.y), (next_node.x, next_node.y), 4)

		for node in graph:
			r = 4
			if node in (graph.start_node, graph.target_node):
				colour = (255, 64, 0)
				r = 6
			elif node in open_set:
				colour = (0, 255, 0)
			elif node in closed_set:
				colour = (0, 128, 255)
			else:
				colour = (80, 80, 80)

			pg.draw.circle(scene, colour, (node.x, node.y), r)

		pg.display.update()
		clock.tick(90)


if __name__ == '__main__':
	pg.init()
	pg.display.set_caption('A* and Dijkstra demo')
	scene = pg.display.set_mode((WIDTH, HEIGHT))
	clock = pg.time.Clock()

	root = tk.Tk()
	root.title('A*/Dijkstra Demo')
	root.config(width=380, height=235, background='#101010')
	root.resizable(False, False)

	stringv_type = tk.StringVar(value='labyrinth')
	radio_btn_labyrinth = tk.Radiobutton(
		root, text='Labyrinth', font='consolas 10', variable=stringv_type, value='labyrinth',
		background='#101010', foreground='white',
		activebackground='#101010', activeforeground='white', selectcolor='#101010'
	)
	radio_btn_maze = tk.Radiobutton(
		root, text='Maze', font='consolas 10', variable=stringv_type, value='maze',
		background='#101010', foreground='white',
		activebackground='#101010', activeforeground='white', selectcolor='#101010'
	)
	radio_btn_graph = tk.Radiobutton(
		root, text='Graph', font='consolas 10', variable=stringv_type, value='graph',
		background='#101010', foreground='white',
		activebackground='#101010', activeforeground='white', selectcolor='#101010'
	)

	btn_generate = tk.Button(root, text='Generate', font='consolas', command=generate_graph)
	btn_solve_astar = tk.Button(root, text='Solve with A*', font='consolas', command=astar)
	btn_solve_dijkstra = tk.Button(root, text='Solve with Dijkstra', font='consolas', command=dijkstra)
	btn_solve_dfs = tk.Button(root, text='Solve with DFS', font='consolas', command=lambda: graph_traversal('dfs'))
	btn_solve_bfs = tk.Button(root, text='Solve with BFS', font='consolas', command=lambda: graph_traversal('bfs'))

	radio_btn_labyrinth.place(width=95, height=32, x=20, y=40, anchor='w')
	radio_btn_maze.place(width=65, height=32, x=115, y=40, anchor='w')
	radio_btn_graph.place(width=70, height=32, x=180, y=40, anchor='w')

	btn_generate.place(width=100, height=32, x=310, y=40, anchor='center')
	btn_solve_astar.place(width=280, height=32, relx=0.5, y=90, anchor='center')
	btn_solve_dijkstra.place(width=280, height=32, relx=0.5, y=125, anchor='center')
	btn_solve_dfs.place(width=280, height=32, relx=0.5, y=160, anchor='center')
	btn_solve_bfs.place(width=280, height=32, relx=0.5, y=195, anchor='center')

	generate_graph()

	root.mainloop()
