# Python drawing
# Author: Sam Barba
# Created 29/10/2018

import random
from time import sleep
import turtle

t = turtle.Turtle()
s = turtle.Screen()

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def reset():
	t.goto(0, 0)
	t.seth(90)
	t.hideturtle()
	t.speed(0)
	t.pensize(1)
	t.color("black")
	t.clear()
	s.colormode(255)
	s.tracer(10)
	s.setup(width=1.0, height=1.0)

def cantor(x=-600.0, y=150, l=1200.0):
	if l < 1: return

	t.pu()
	t.goto(x, y)
	t.pd()
	t.seth(0)
	t.fd(l)
	sleep(0.05)
	cantor(x, y - 40, l / 3)
	cantor(x + l * 2 / 3, y - 40, l / 3)

def dragon(lvl=14, size=3, h=45):
	if lvl:
		t.rt(h)
		dragon(lvl - 1, size)
		t.lt(h * 2)
		dragon(lvl - 1, size, -45)
		t.rt(h)
	else:
		t.fd(size)

# Sierpinski triangle (t.rt(90) before)
def sierp(size=400.0, lvl=6):
	if lvl == 0:
		for i in range(3):
			t.fd(size)
			t.lt(120)
	else:
		t.begin_fill()
		sierp(size / 2, lvl - 1)
		t.fd(size / 2)
		sierp(size / 2, lvl - 1)
		t.bk(size / 2)
		t.lt(60)
		t.fd(size / 2)
		t.rt(60)
		sierp(size / 2, lvl - 1)
		t.lt(60)
		t.bk(size / 2)
		t.rt(60)
		t.end_fill()

# Koch snowflake (t.goto(-600, 0) and t.rt(90) before)
def koch(size=1000.0, lvl=6):
	if lvl:
		for i in [60, -120, 60, 0]:
			koch(size / 3, lvl - 1)
			t.lt(i)
	else:
		t.fd(size)

def tSquare(x=0.0, y=0.0, size=400.0):
	if size < 4: return

	t.pu()
	t.goto(x - size / 2, y - size / 2)
	t.pd()
	t.seth(90)
	t.begin_fill()
	for i in range(4):
		t.fd(size)
		t.rt(90)
	t.end_fill()

	tSquare(x - size / 2, y - size / 2, size / 2)
	tSquare(x - size / 2, y + size / 2, size / 2)
	tSquare(x + size / 2, y - size / 2, size / 2)
	tSquare(x + size / 2, y + size / 2, size / 2)

# Fractal tree (angle = 20/30/45/90)
def tree(angle, size=60):
	if size > 5:
		t.fd(size)
		t.rt(angle)
		tree(size - 5, angle)
		t.lt(angle * 2)
		tree(size - 5, angle)
		t.rt(angle)
		t.bk(size)

# Spirals:
def spirals():
	for i in [45, 51, 60, 72, 90, 103, 120, 144, 168, 179]:
		spiral(i)
		sleep(1)
		reset()

# Spiral:
def spiral(angle):
	for i in range(300):
		t.fd(i)
		t.lt(angle)

def drawSquare(size, angle):
	t.seth(angle)
	t.fillcolor((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
	t.begin_fill()
	for i in range(4):
		t.fd(size)
		t.lt(90)
	t.end_fill()

def hilbert(size=6, level=6, angle=90):
	if level == 0: return

	t.rt(angle)
	hilbert(size, level - 1, -angle)
	t.fd(size)
	t.lt(angle)
	hilbert(size, level - 1, angle)
	t.fd(size)
	hilbert(size, level - 1, angle)
	t.lt(angle)
	t.fd(size)
	hilbert(size, level - 1, -angle)
	t.rt(angle)

def randWalk(bound, step):
	t.dot(10000, "black")
	t.pu()
	t.goto(bound, bound)
	t.pd()
	t.seth(270)
	t.color("red")
	for i in range(4):
		t.fd(bound * 2)
		t.rt(90)
	t.pu()
	t.goto(0, 0)
	t.color("black")

	path = [t.pos()]

	while True:
		angle = random.choice([-90, 0, 90, 180])
		t.seth(angle)
		t.fd(step)
		path.append(t.pos())
		if abs(t.pos()[0]) > bound or abs(t.pos()[1]) > bound:
			break

	t.pd()
	for idx, node in enumerate(path):
		t.goto(node)
		c = round(mapRange(idx, 0, len(path), 30, 255))
		t.color((c, c, c))

def mapRange(x, fromLo, fromHi, toLo, toHi):
	return (x - fromLo) * (toHi - toLo) / (fromHi - fromLo) + toLo

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

reset()

#spirals()
#sleep(2)

#reset()
#size, angle = 300, 0
#while size > 10:
#	drawSquare(size, angle)
#	size -= 0.2
#	angle += 3
#sleep(2)

#reset()
#hilbert()
#sleep(2)

#tSquare()
#sleep(2)
#reset()

dragon()
sleep(2)

randWalk(400, 2)

input("Press Enter to exit")
