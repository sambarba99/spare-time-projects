"""
A* and Dijkstra demo

Author: Sam Barba
Created 20/09/2021
"""

from daedalus import Daedalus
from graph_gen import GraphGen
import numpy as np
import pygame as pg
import sys
from time import sleep
import tkinter as tk

CELL_SIZE = 6  # For maze mode

maze_mode = True  # False means normal graph (vertices/edges) mode
maze_generator = Daedalus()
graph_generator = GraphGen(max_x=maze_generator.cols * CELL_SIZE, max_y=maze_generator.rows * CELL_SIZE)
graph = start_vertex = target_vertex = path = None

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def generate_and_draw_graph():
	global graph, start_vertex, target_vertex, path

	if maze_mode:
		graph = maze_generator.make_maze()
		# Start and target are top-left and bottom-right, respectively
		start_vertex = graph[0][0]
		target_vertex = graph[maze_generator.rows - 1][maze_generator.cols - 1]
	else:
		graph = graph_generator.make_graph()
		coords = np.array(list(zip([v.x for v in graph], [v.y for v in graph])))

		# Start is top-left-most vertex; target is bottom-right-most vertex
		distances_from_top_left = np.linalg.norm(coords, axis=1)
		distances_from_bottom_right = np.linalg.norm(
			coords - np.array([graph_generator.max_x, graph_generator.max_y]),
			axis=1
		)
		start_vertex = graph[np.argmin(distances_from_top_left)]
		target_vertex = graph[np.argmin(distances_from_bottom_right)]

	path = None
	draw()

def a_star():
	open_set, closed_set = [start_vertex], []

	while open_set:
		# Sort by f_cost then by h_cost
		cheapest_vertex = min(open_set, key=lambda v: (v.get_f_cost(), v.h_cost))

		if cheapest_vertex is target_vertex:
			retrace_path()
			draw()
			return

		open_set.remove(cheapest_vertex)
		closed_set.append(cheapest_vertex)

		if maze_mode: neighbours = cheapest_vertex.get_neighbours(graph, maze_generation=False)
		else: neighbours = cheapest_vertex.neighbours

		for n in neighbours:
			if n in closed_set: continue

			cost_move_to_n = cheapest_vertex.g_cost + cheapest_vertex.dist(n)
			if cost_move_to_n < n.g_cost or n not in open_set:
				n.g_cost = cost_move_to_n
				n.h_cost = n.dist(target_vertex)
				n.parent = cheapest_vertex

				if n not in open_set:
					open_set.append(n)

def dijkstra():
	"""Generates Shortest Path Tree"""

	unvisited = [vertex for row in graph for vertex in row if not vertex.is_wall] \
		if maze_mode else [v for v in graph]

	# Costs nothing to get from start to start (start_vertex parent will always be None)
	start_vertex.cost = 0

	while unvisited:
		cheapest_vertex = min(unvisited, key=lambda v: v.cost)

		if cheapest_vertex is target_vertex:
			# Stop generating Shortest Path Tree
			# (only need path from start_vertex to target_vertex)
			break

		neighbours = cheapest_vertex.get_neighbours(graph, maze_generation=False) \
			if maze_mode else cheapest_vertex.neighbours

		for n in neighbours:
			step = 1 if maze_mode else cheapest_vertex.dist(n)

			if cheapest_vertex.cost + step < n.cost:
				n.cost = cheapest_vertex.cost + step
				n.parent = cheapest_vertex

		# Cheapest vertex has now been visited
		unvisited.remove(cheapest_vertex)

	retrace_path()
	draw()

def retrace_path():
	global path

	# Trace back from end
	current = target_vertex
	retraced_path = [current]

	while current != start_vertex:
		current = current.parent
		retraced_path.append(current)

	path = retraced_path[::-1]

def draw():
	scene.fill((0, 0, 0))

	if maze_mode:
		for y in range(maze_generator.rows):
			for x in range(maze_generator.cols):
				c = (0, 0, 0) if graph[y][x].is_wall else (80, 80, 80)

				pg.draw.rect(scene, c, pg.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
	else:
		# Draw edges then vertices on top
		for v in graph:
			for neighbour in v.neighbours:
				pg.draw.line(scene, (160, 160, 160), (v.x, v.y), (neighbour.x, neighbour.y))

		for v in graph:
			pg.draw.circle(scene, (255, 0, 0), (v.x, v.y), 4)

		# Start and target
		pg.draw.circle(scene, (0, 80, 255), (start_vertex.x, start_vertex.y), 8)
		pg.draw.circle(scene, (0, 80, 255), (target_vertex.x, target_vertex.y), 8)

	pg.display.update()

	if path is None: return

	if maze_mode:
		time_interval = 5 / len(path)  # Want drawing to last around 5s

		for v in path:
			for event in pg.event.get():
				if event.type == pg.QUIT:
					sys.exit(0)
			pg.draw.rect(scene, (220, 0, 0), pg.Rect(v.x * CELL_SIZE, v.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
			sleep(time_interval)
			pg.display.update()
	else:
		# Draw edges then vertices on top
		for v in graph:
			for neighbour in v.neighbours:
				pg.draw.line(scene, (160, 160, 160), (v.x, v.y), (neighbour.x, neighbour.y))

		for vertex, next_vertex in zip(path[:-1], path[1:]):
			pg.draw.line(scene, (0, 255, 0), (vertex.x, vertex.y), (next_vertex.x, next_vertex.y), 4)

		for v in graph:
			pg.draw.circle(scene, (255, 0, 0), (v.x, v.y), 4)

		# Start and target
		pg.draw.circle(scene, (0, 80, 255), (start_vertex.x, start_vertex.y), 8)
		pg.draw.circle(scene, (0, 80, 255), (target_vertex.x, target_vertex.y), 8)

	pg.display.update()

def toggle_maze_mode():
	global maze_mode
	maze_mode = not maze_mode
	generate_and_draw_graph()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	pg.init()
	pg.display.set_caption('A* and Dijkstra demo')
	scene = pg.display.set_mode((maze_generator.cols * CELL_SIZE, maze_generator.rows * CELL_SIZE))

	generate_and_draw_graph()

	root = tk.Tk()
	root.title('A*/Dijkstra Demo')
	root.config(width=350, height=230, bg='#000024')

	btn_generate_graph = tk.Button(root, text='Generate graph', font='consolas',
		command=lambda: generate_and_draw_graph())
	btn_solve_a_star = tk.Button(root, text='Solve with A*', font='consolas',
		command=lambda: a_star())
	btn_solve_dijkstra = tk.Button(root, text='Solve with Dijkstra', font='consolas',
		command=lambda: dijkstra())
	btn_toggle_maze_mode = tk.Button(root, text='Toggle maze/graph mode', font='consolas',
		command=lambda: toggle_maze_mode())

	btn_generate_graph.place(relwidth=0.8, relheight=0.16, relx=0.5, rely=0.2, anchor='center')
	btn_solve_a_star.place(relwidth=0.8, relheight=0.16, relx=0.5, rely=0.4, anchor='center')
	btn_solve_dijkstra.place(relwidth=0.8, relheight=0.16, relx=0.5, rely=0.6, anchor='center')
	btn_toggle_maze_mode.place(relwidth=0.8, relheight=0.16, relx=0.5, rely=0.8, anchor='center')

	root.mainloop()
