# Pen for drawing.py
# Author: Sam Barba
# Created 19/02/2022

from math import radians, sin, cos
import pygame as pg

class Pen:
	def __init__(self, scene, x, y, heading):
		self.scene = scene
		self.x = x
		self.y = y
		self.heading = heading

	def go_to(self, x, y):
		self.x = x
		self.y = y

	def turn(self, theta):
		self.heading += theta
		self.heading = round(self.heading)

	def move(self, length, draw=True, colour=(220, 220, 220)):
		end_x = length * cos(radians(self.heading)) + self.x
		end_y = length * sin(radians(self.heading)) + self.y

		if draw:
			pg.draw.line(self.scene, colour, (self.x, self.y), (end_x, end_y))
			pg.display.flip()

		self.x = end_x
		self.y = end_y

	def pos(self):
		return self.x, self.y
