# A* and Dijkstra demo
# Author: Sam Barba
# Created 20/09/2021

from daedalus import Daedalus
import pygame as pg
import sys
from time import sleep
import tkinter as tk

ROWS = 59
COLS = 99
CELL_SIZE = 12

maze_generator = Daedalus(ROWS, COLS)
maze = None
start_vertex = None
target_vertex = None
path = None

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def generate_and_draw_maze():
	global maze, start_vertex, target_vertex, path

	maze = maze_generator.make_maze()
	start_vertex = maze[0][0]
	target_vertex = maze[ROWS - 1][COLS - 1]
	path = None

	draw()

def a_star():
	global maze, start_vertex, target_vertex

	open_set, closed_set = [start_vertex], []

	while open_set:
		cheapest_vertex = min(open_set, key=lambda v: (v.get_f_cost(), v.h_cost))

		if cheapest_vertex == target_vertex:
			retrace_path()
			draw()
			return

		open_set.remove(cheapest_vertex)
		closed_set.append(cheapest_vertex)

		neighbours = cheapest_vertex.get_neighbours(maze, False)
		for n in neighbours:
			if n in closed_set: continue

			cost_move_to_n = cheapest_vertex.g_cost + manhattan_dist(cheapest_vertex, n)
			if cost_move_to_n < n.g_cost or n not in open_set:
				n.g_cost = cost_move_to_n
				n.h_cost = manhattan_dist(n, target_vertex)
				n.parent_vertex = cheapest_vertex

				if n not in open_set:
					open_set.append(n)

def manhattan_dist(a, b):
	return abs(a.x - b.x) + abs(a.y - b.y)

# Dijkstra's algorithm for Shortest Path Tree
def dijkstra():
	global maze, start_vertex, target_vertex

	unvisited = [vertex for row in maze for vertex in row if not vertex.is_wall]

	# Costs nothing to get from start to start (start_vertex parent will always be None)
	start_vertex.cost = 0

	while unvisited:
		cheapest_vertex = min(unvisited, key=lambda v: v.cost)

		neighbours = cheapest_vertex.get_neighbours(maze, False)
		for n in neighbours:
			# Adjust cost and parent (weight between vertices = 1, i.e. 1 step needed)
			if cheapest_vertex.cost + 1 < n.cost:
				n.cost = cheapest_vertex.cost + 1
				n.parent_vertex = cheapest_vertex

		# Cheapest vertex has now been visited
		unvisited.remove(cheapest_vertex)

	retrace_path()
	draw()

def retrace_path():
	global start_vertex, target_vertex, path

	# Trace back from end
	current = target_vertex
	retraced_path = [current]

	while current != start_vertex:
		current = current.parent_vertex
		retraced_path.append(current)

	path = retraced_path[::-1]

def draw():
	global maze, path

	for y in range(ROWS):
		for x in range(COLS):
			c = (0, 0, 0) if maze[y][x].is_wall else (80, 80, 80)

			pg.draw.rect(scene, c, pg.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

	pg.display.update()

	if path is None: return

	time_interval = 5 / len(path)  # Want drawing to last around 5s

	for v in path:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				sys.exit(0)
		pg.draw.rect(scene, (220, 0, 0), pg.Rect(v.x * CELL_SIZE, v.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
		sleep(time_interval)
		pg.display.update()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

pg.init()
pg.display.set_caption("A* and Dijkstra demo")
scene = pg.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE))

generate_and_draw_maze()

root = tk.Tk()
root.title("A*/Dijkstra Maze Solver")
root.configure(width=300, height=200, bg="#141414")

frame = tk.Frame(root, bg="#0080ff")
frame.place(relwidth=0.9, relheight=0.9, relx=0.5, rely=0.5, anchor="center")

btn_generate_maze = tk.Button(frame, text="Generate maze", font="consolas", command=lambda: generate_and_draw_maze())
btn_solve_a_star = tk.Button(frame, text="Solve with A*", font="consolas", command=lambda: a_star())
btn_solve_dijkstra = tk.Button(frame, text="Solve with Dijkstra", font="consolas", command=lambda: dijkstra())
btn_generate_maze.place(relwidth=0.8, relheight=0.2, relx=0.5, rely=0.25, anchor="center")
btn_solve_a_star.place(relwidth=0.8, relheight=0.2, relx=0.5, rely=0.5, anchor="center")
btn_solve_dijkstra.place(relwidth=0.8, relheight=0.2, relx=0.5, rely=0.75, anchor="center")

root.mainloop()
