# Golden Ratio visualiser
# Author: Sam Barba
# Created 21/09/2021

from math import pi, sin, cos
import pygame as pg
import tkinter as tk

SIZE = 900
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2  # ~0.61803

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def set_turn_ratio_and_draw(ratio=None):
	if ratio is None: ratio = slider.get()
	if ratio == 0.618: ratio = GOLDEN_RATIO

	turn_ratio = ratio
	slider.set(ratio)
	radius, angle = 1, 0
	scene.fill((0, 0, 0))

	while radius < SIZE * 0.45:
		x = radius * cos(angle) + SIZE / 2
		y = radius * sin(angle) + SIZE / 2
		pg.draw.circle(scene, (255, 160, 0), (x, y), 1)
		angle = (angle + 2 * pi * turn_ratio) % (2 * pi)
		radius += 0.1

	font = pg.font.SysFont("consolas", 18)
	turn_ratio_lbl = font.render(f"Turn ratio: {turn_ratio:.3f}", True, (220, 220, 220))
	scene.blit(turn_ratio_lbl, (10, 10))

	pg.display.update()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
	pg.init()
	pg.display.set_caption("Golden ratio visualiser")
	scene = pg.display.set_mode((SIZE, SIZE))

	root = tk.Tk()
	root.title("Golden Ratio Visualiser")
	root.configure(width=300, height=200, bg="#141414")

	frame = tk.Frame(root, bg="#0080ff")
	frame.place(relwidth=0.9, relheight=0.9, relx=0.5, rely=0.5, anchor="center")

	select_ratio_lbl = tk.Label(frame, text="Select a turn ratio:", font="consolas", bg="#0080ff")
	select_ratio_lbl.place(relwidth=0.8, relheight=0.2, relx=0.5, rely=0.18, anchor="center")

	slider = tk.Scale(frame, from_=0.5, to=1, resolution=0.001, orient="horizontal", font="consolas",
		command=lambda l: set_turn_ratio_and_draw())
	slider.place(relwidth=0.8, relheight=0.3, relx=0.5, rely=0.44, anchor="center")

	btn_set_to_golden_ratio = tk.Button(frame, text="Set to golden ratio", font="consolas",
		command=lambda: set_turn_ratio_and_draw(GOLDEN_RATIO))
	btn_set_to_golden_ratio.place(relwidth=0.8, relheight=0.2, relx=0.5, rely=0.77, anchor="center")

	set_turn_ratio_and_draw(0.5)

	root.mainloop()
