# Minimum Spanning Tree demo
# Author: Sam Barba
# Created 17/09/2021

# Left-click: add a vertex
# Right-click: reset graph

import pygame as pg
import random
import sys

SIZE = 800
FPS = 20

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Prim's algorithm
def mst(graph):
	out_tree = graph[:]  # Initially set all vertices as out of tree
	in_tree = []
	mst_parents = [None] * len(graph)

	in_tree.append(out_tree.pop(0))  # Vertex 0 (arbitrary start) is first in tree

	while out_tree:
		nearest_in = in_tree[0]
		nearest_out = out_tree[0]
		min_dist = euclidean_dist(nearest_in, nearest_out)

		# Find the nearest outside vertex to tree
		for v_in in in_tree:
			for v_out in out_tree:
				dist = euclidean_dist(v_in, v_out)

				if dist < min_dist:
					min_dist = dist
					nearest_out = v_out
					nearest_in = v_in

		mst_parents[nearest_out["idx"]] = nearest_in["idx"]

		in_tree.append(nearest_out)
		out_tree.remove(nearest_out)

	return mst_parents

def euclidean_dist(a, b):
	# Ignore square root for faster execution
	return (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2

def draw_mst(graph):
	if not graph: return

	scene.fill((20, 20, 20))
	mst_parents = mst(graph)

	for idx, v in enumerate(graph[1:], start=1):  # Start from 1 because mstParents[0] is None
		start = (v["x"], v["y"])
		end = (graph[mst_parents[idx]]["x"], graph[mst_parents[idx]]["y"])
		pg.draw.line(scene, (220, 220, 220), start, end)

	for v in graph:
		pg.draw.circle(scene, (230, 20, 20), (v["x"], v["y"]), 5)

	pg.display.update()

def move_points(graph):
	for v in graph:
		v["x"] += v["x-vel"]
		v["y"] += v["y-vel"]

		while v["x"] < 5 or v["x"] > SIZE - 5:
			v["x-vel"] = -v["x-vel"]
			v["x"] += v["x-vel"]
		while v["y"] < 5 or v["y"] > SIZE - 5:
			v["y-vel"] = -v["y-vel"]
			v["y"] += v["y-vel"]

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

graph = []

pg.init()
pg.display.set_caption("Minimum Spanning Tree")
scene = pg.display.set_mode((SIZE, SIZE))
scene.fill((20, 20, 20))
pg.display.update()
clock = pg.time.Clock()

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)
		elif event.type == pg.MOUSEBUTTONDOWN:
			if event.button == 1:  # Left-click
				if len(graph) == 30:
					print("Size limit reached")
					continue

				x, y = event.pos
				# Constrain x and y to range [5, SIZE - 5]
				x = max(min(x, SIZE - 5), 5)
				y = max(min(y, SIZE - 5), 5)

				x_vel, y_vel = random.uniform(-3, 3), random.uniform(-3, 3)
				vertex = {"idx": len(graph), "x": x, "y": y, "x-vel": x_vel, "y-vel": y_vel}
				graph.append(vertex)

			elif event.button == 3:  # Right-click
				graph = []
				scene.fill((20, 20, 20))
				pg.display.update()

	draw_mst(graph)
	move_points(graph)
	clock.tick(FPS)
