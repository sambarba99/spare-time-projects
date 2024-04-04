"""
A* and Dijkstra demo

Author: Sam Barba
Created 20/09/2021
"""

import sys
import tkinter as tk

import numpy as np
import pygame as pg

from daedalus import make_maze
from graph_gen import make_graph


# For maze mode
ROWS = 99
COLS = 149
CELL_SIZE = 7

maze_mode = True  # False means normal graph (nodes/edges) mode
graph = start_node = target_node = path = None


def generate_and_draw_graph():
	global graph, start_node, target_node, path

	if maze_mode:
		graph = make_maze(ROWS, COLS)
		# Start and target are top-left and bottom-right, respectively
		start_node = graph[0][0]
		target_node = graph[-1][-1]
	else:
		graph = make_graph(x_max=COLS * CELL_SIZE, y_max=ROWS * CELL_SIZE)
		np_coords = np.array(list(zip([n.x for n in graph], [n.y for n in graph])))

		# Start is top-left-most node; target is node furthest from this
		distances_from_top_left = np.linalg.norm(np_coords, axis=1)
		start_node = graph[distances_from_top_left.argmin()]
		target_node = graph[distances_from_top_left.argmax()]

	path = None
	draw()


def a_star():
	open_set, closed_set = {start_node}, set()

	while open_set:
		# Sort by f_cost then by h_cost
		cheapest_node = min(open_set, key=lambda n: (n.get_f_cost(), n.h_cost))

		if cheapest_node is target_node:
			retrace_path()
			draw()
			return

		open_set.remove(cheapest_node)
		closed_set.add(cheapest_node)

		neighbours = cheapest_node.get_neighbours(graph, maze_generation=False) \
			if maze_mode \
			else cheapest_node.neighbours

		for n in neighbours:
			if n in closed_set: continue

			cost_move_to_n = cheapest_node.g_cost + cheapest_node.dist(n)
			if cost_move_to_n < n.g_cost or n not in open_set:
				n.g_cost = cost_move_to_n
				n.h_cost = n.dist(target_node)
				n.parent = cheapest_node
				open_set.add(n)


def dijkstra():
	"""Generates Shortest Path Tree"""

	unvisited = {node for row in graph for node in row if not node.is_wall} \
		if maze_mode \
		else {n for n in graph}

	# Costs nothing to get from start to start (start_node parent will always be None)
	start_node.cost = 0

	while unvisited:
		cheapest_node = min(unvisited, key=lambda node: node.cost)

		if cheapest_node is target_node:
			# Stop generating Shortest Path Tree
			# (only need path from start_node to target_node)
			break

		neighbours = cheapest_node.get_neighbours(graph, maze_generation=False) \
			if maze_mode \
			else cheapest_node.neighbours

		for n in neighbours:
			step = 1 if maze_mode else cheapest_node.dist(n)

			if cheapest_node.cost + step < n.cost:
				n.cost = cheapest_node.cost + step
				n.parent = cheapest_node

		# Cheapest node has now been visited
		unvisited.remove(cheapest_node)

	retrace_path()
	draw()


def retrace_path():
	global path

	# Trace back from end
	current = target_node
	retraced_path = [current]

	while current != start_node:
		current = current.parent
		retraced_path.append(current)

	path = retraced_path[::-1]


def draw():
	scene.fill('black')

	if maze_mode:
		for y in range(ROWS):
			for x in range(COLS):
				c = (0, 0, 0) if graph[y][x].is_wall else (80, 80, 80)

				pg.draw.rect(scene, c, pg.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

		if path:
			for node in path:
				pg.draw.rect(scene, (220, 0, 0), pg.Rect(node.x * CELL_SIZE, node.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
				pg.time.delay(2)
				pg.display.update()
	else:
		# Draw edges then nodes on top
		for node in graph:
			for neighbour in node.neighbours:
				pg.draw.line(scene, (160, 160, 160), (node.x, node.y), (neighbour.x, neighbour.y))

		if path:
			for node, next_node in zip(path[:-1], path[1:]):
				pg.draw.line(scene, (0, 255, 0), (node.x, node.y), (next_node.x, next_node.y), 4)

		for node in graph:
			pg.draw.circle(scene, (255, 0, 0), (node.x, node.y), 3)

		# Start and target
		pg.draw.circle(scene, (0, 80, 255), (start_node.x, start_node.y), 6)
		pg.draw.circle(scene, (0, 80, 255), (target_node.x, target_node.y), 6)

	pg.display.update()


def toggle_maze_mode():
	global maze_mode
	maze_mode = not maze_mode
	generate_and_draw_graph()


if __name__ == '__main__':
	pg.init()
	pg.display.set_caption('A* and Dijkstra demo')
	scene = pg.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE))

	generate_and_draw_graph()

	root = tk.Tk()
	root.title('A*/Dijkstra Demo')
	root.config(width=350, height=230, background='#101010')
	root.resizable(False, False)

	btn_generate_graph = tk.Button(root, text='Generate graph', font='consolas',
		command=lambda: generate_and_draw_graph())
	btn_solve_a_star = tk.Button(root, text='Solve with A*', font='consolas',
		command=lambda: a_star())
	btn_solve_dijkstra = tk.Button(root, text='Solve with Dijkstra', font='consolas',
		command=lambda: dijkstra())
	btn_toggle_maze_mode = tk.Button(root, text='Toggle maze/graph mode', font='consolas',
		command=lambda: toggle_maze_mode())

	btn_generate_graph.place(width=280, height=32, relx=0.5, y=46, anchor='center')
	btn_solve_a_star.place(width=280, height=32, relx=0.5, y=92, anchor='center')
	btn_solve_dijkstra.place(width=280, height=32, relx=0.5, y=138, anchor='center')
	btn_toggle_maze_mode.place(width=280, height=32, relx=0.5, y=184, anchor='center')

	root.mainloop()
