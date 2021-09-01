# Drawing with Fourier Transforms
# Author: Sam Barba
# Created 23/09/2021

# Right click: enter drawing mode (then left click drag to draw)
# 1 - 3: change number of epicycles (change drawing precision)
# P: draw pi symbol

from complex import *
from presets import *
import math
import pygame as pg
import random
import sys

SIZE = 800
FPS = 120

numEpicycles = 3
userDrawing = False
leftButtonDown = False
userCoords = []
fourier = []
epicycleCircles = [None] * numEpicycles
epicycleLines = [None] * numEpicycles
positionsToDraw = []
time = 0
paused = False

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def setFourier(coords):
	complexValues = [getComplexBetween(*coords[i], *coords[i + 1]) for i in range(len(coords) - 1)]

	first, last = coords[0], coords[len(coords) - 1]
	complexValues.append(getComplexBetween(*last, *first))

	# 'Flatten' list
	complexValues = sum(complexValues, [])

	global fourier, time

	fourier = dft(complexValues)
	time = 0

# Bresenham's algorithm
def getComplexBetween(re1, im1, re2, im2):
	dRe = abs(re2 - re1)
	dIm = -abs(im2 - im1)
	sRe = 1 if re1 < re2 else -1
	sIm = 1 if im1 < im2 else -1
	err = dRe + dIm

	result = []

	while True:
		result.append(Complex(re1, im1))

		if re1 == re2 and im1 == im2: return result

		e2 = 2 * err
		if e2 >= dIm:
			err += dIm
			re1 += sRe
		if e2 <= dRe:
			err += dRe
			im1 += sIm

# Discrete Fourier Transform
def dft(complexValues):
	n = len(complexValues)
	result = []

	for i in range(n):
		s = Complex(0, 0)

		for j in range(n):
			phi = 2 * math.pi * i * j / n
			c = Complex(math.cos(phi), -math.sin(phi))
			s = s.add(complexValues[j].mult(c))

		s.re /= n
		s.im /= n
		s.freq = i

		result.append(s)

	return sorted(result, key = lambda c: c.getAmp(), reverse = True)

def draw(scene):
	global time, fourier, positionsToDraw, epicycleCircles, epicycleLines

	scene.fill((0, 0, 0))

	epicycleCoords = epicycles(SIZE // 2, SIZE // 2)
	x, y = map(round, epicycleCoords)
	positionsToDraw.append([x, y])
	positionsToDraw = positionsToDraw[-1000:]

	for i in range(len(positionsToDraw) - 1):
		pg.draw.line(scene, (255, 0, 0), positionsToDraw[i], positionsToDraw[i + 1])

	for circle in epicycleCircles:
		if circle == None:
			continue
		pg.draw.circle(scene, *circle)
	for line in epicycleLines:
		if line == None:
			continue
		pg.draw.line(scene, *line)

	pg.display.flip()

	time += 2 * math.pi / len(fourier)
	time %= 2 * math.pi

def epicycles(x, y):
	global numEpicycles, fourier, epicycleCircles, epicycleLines

	lim = min(numEpicycles, len(fourier))

	for i in range(lim):
		prevX, prevY = x, y
		r = fourier[i].getAmp()
		f = fourier[i].freq
		phase = fourier[i].getPhase()

		x += r * math.cos(f * time + phase)
		y += r * math.sin(f * time + phase)

		# Colour, centre, radius, border width
		epicycleCircles[i] = (90, 90, 90), (prevX, prevY), r, 1
		# Colour, start pos, end pos
		epicycleLines[i] = (255, 255, 255), (prevX, prevY), (x, y)

	return x, y

def refresh(scene):
	global fourier, epicycleCircles, epicycleLines, numEpicycles, positionsToDraw

	epicycleCircles = [None] * numEpicycles
	epicycleLines = [None] * numEpicycles
	positionsToDraw = []
	scene.fill((0, 0, 0))
	pg.display.flip()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

pg.init()
pg.display.set_caption("Fourier")
scene = pg.display.set_mode((SIZE, SIZE))
clock = pg.time.Clock()

print("\nStarting...")
setFourier(PI)

while True:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			sys.exit(0)

		elif event.type == pg.MOUSEBUTTONDOWN:
			if event.button == 1:
				leftButtonDown = True

			if event.button == 1 and userDrawing:
				x, y = event.pos
				userCoords.append([x - SIZE // 2, y - SIZE // 2])
				scene.set_at((x, y), (255, 0, 0))
				pg.display.flip()

			elif event.button == 3:
				userDrawing = not userDrawing
				paused = userDrawing
				refresh(scene)

				if userDrawing:
					print("Starting drawing")
					userCoords = []
				else:
					print("Finished drawing")
					if len(userCoords) > 1:
						setFourier(userCoords)

		elif event.type == pg.MOUSEBUTTONUP:
			if event.button == 1:
				leftButtonDown = False

		elif event.type == pg.MOUSEMOTION:
			# Draw by holding down left button and dragging mouse
			if userDrawing and leftButtonDown:
				x, y = event.pos
				userCoords.append([x - SIZE // 2, y - SIZE // 2])
				scene.set_at((x, y), (255, 0, 0))
				pg.display.flip()

		elif event.type == pg.KEYDOWN:
			if event.key == pg.K_1 and not userDrawing:
				print("Changing epicycles to 3")
				numEpicycles = 3
				refresh(scene)
			elif event.key == pg.K_2 and not userDrawing:
				print("Changing epicycles to 8")
				numEpicycles = 8
				refresh(scene)
			elif event.key == pg.K_3 and not userDrawing:
				print("Changing epicycles to 200")
				numEpicycles = 200
				refresh(scene)
			elif event.key == pg.K_p:
				print("Setting 'pi' preset")
				refresh(scene)
				setFourier(PI)
				userDrawing = paused = False
			elif event.key == pg.K_r: # Just random coordinates around centre
				print("Randomising coordinates")
				allCoords = [[x, y] for x in range(-100, 101) for y in range(-100, 101)]
				refresh(scene)
				setFourier(random.sample(allCoords, 6))
				userDrawing = paused = False

	if not paused: draw(scene)

	clock.tick(FPS)
