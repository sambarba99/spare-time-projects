# Golden Ratio visualiser
# Author: Sam Barba
# Created 21/09/2021

import math
import pygame as pg
import sys
from time import sleep

TURN_RATIO = (5 ** 0.5 - 1) / 2
SIZE = 900

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

pg.init()
pg.display.set_caption("Golden ratio visualiser")
scene = pg.display.set_mode((SIZE, SIZE))

radius, angle = 1, 0

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)

	if radius > SIZE * 0.48:
		sleep(1)
		radius, angle = 1, 0
		scene.fill((0, 0, 0))

	x = math.cos(angle) * radius + SIZE / 2
	y = math.sin(angle) * radius + SIZE / 2

	pg.draw.circle(scene, (255, 160, 0), (x, y), 1)
	pg.display.flip()

	angle += 2 * math.pi * TURN_RATIO
	angle %= 2 * math.pi
	radius += 0.1
