# A* and Dijkstra demo
# Author: Sam Barba
# Created 20/09/2021

from daedalus import Daedalus
import pygame as pg
from time import sleep

ROWS = 49
COLS = 89
CELL_SIZE = 15

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def a_star(maze, start_vertex, target_vertex):
	open_set, closed_set = [start_vertex], []

	while open_set:
		cheapest_vertex = min(open_set, key=lambda v: (v.get_f_cost(), v.h_cost))

		if cheapest_vertex == target_vertex:
			return retrace_path(target_vertex, start_vertex)

		open_set.remove(cheapest_vertex)
		closed_set.append(cheapest_vertex)

		neighbours = cheapest_vertex.get_neighbours(maze, False)
		for n in neighbours:
			if n in closed_set: continue

			cost_move_to_n = cheapest_vertex.g_cost + dist(cheapest_vertex, n)
			if cost_move_to_n < n.g_cost or n not in open_set:
				n.g_cost = cost_move_to_n
				n.h_cost = dist(n, target_vertex)
				n.parent_vertex = cheapest_vertex

				if n not in open_set:
					open_set.append(n)

# Manhattan distance
def dist(a, b):
	return abs(a.x - b.x) + abs(a.y - b.y)

# Dijkstra's algorithm for Shortest Path Tree
def dijkstra(maze, start_vertex, target_vertex):
	unvisited = [vertex for row in maze for vertex in row if not vertex.is_wall]

	# Costs nothing to get from start to start (startVertex parent will always be None)
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

	return retrace_path(target_vertex, start_vertex)

def retrace_path(target_vertex, start_vertex):
	# Trace back from end
	current = target_vertex
	path = [current]

	while current != start_vertex:
		current = current.parent_vertex
		path.append(current)

	return path[::-1]

def draw(maze, path):
	for y in range(ROWS):
		for x in range(COLS):
			c = (0, 0, 0) if maze[y][x].is_wall else (80, 80, 80)

			pg.draw.rect(scene, c, pg.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

	pg.display.update()

	sleep(1)
	for v in path:
		pg.draw.rect(scene, (220, 0, 0), pg.Rect(v.x * CELL_SIZE, v.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
		sleep(0.01)
		pg.display.update()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

maze_generator = Daedalus(ROWS, COLS)

pg.init()
pg.display.set_caption("A* and Dijkstra demo")
scene = pg.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE))

while True:
	maze = maze_generator.make_maze()

	start_vertex = maze[0][0]
	target_vertex = maze[ROWS - 1][COLS - 1]

	path = a_star(maze, start_vertex, target_vertex)
	#path = dijkstra(maze, start_vertex, target_vertex)

	draw(maze, path)
	sleep(2)
