# Digit recognition neural network demo
# Author: Sam Barba
# Created 20/10/2021

# 1 row in data (MNIST dataset) contains 784 pixel values (i.e. 28*28 image) from 0-255, and a class label (0-9)

import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
import numpy as np
import pygame as pg
import random
import time

DRAWING_SIZE = 500

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Split file data into train/test
def extractData(data, trainTestRatio=0.5):
	data = [row.strip("\n").split() for row in data]

	random.shuffle(data)
	data = np.array(data).astype(float)

	split = int(len(data) * trainTestRatio)

	trainingSet, testSet = data[:split], data[split:]

	xTrain = trainingSet[:,:-1] / 255
	yTrain = trainingSet[:,-1].astype(int)
	xTest = testSet[:,:-1] / 255
	yTest = testSet[:,-1].astype(int)

	yTrain1 = np.zeros((len(yTrain), 10))
	yTrain1[np.arange(len(yTrain)), yTrain] = 1
	yTest1 = np.zeros((len(yTest), 10))
	yTest1[np.arange(len(yTest)), yTest] = 1

	yTrain, yTest = yTrain1, yTest1
	
	return xTrain, yTrain, xTest, yTest

def confusionMatrix(predictions, actual):
	predictions = np.argmax(predictions, axis=1)
	actual = np.argmax(actual, axis=1)
	numClasses = len(np.unique(actual))
	confMat = np.zeros((numClasses, numClasses)).astype(int)

	for a, p in zip(actual, predictions):
		confMat[a, p] += 1

	accuracy = np.trace(confMat) / confMat.sum()
	return confMat, accuracy

def plotMatrix(isTraining, confMat, accuracy):
	fig, ax = plt.subplots(figsize=(6, 7))
	ax.matshow(confMat, cmap=plt.cm.Blues, alpha=0.7)
	ax.xaxis.set_ticks_position("bottom")
	for i in range(confMat.shape[0]):
		for j in range(confMat.shape[1]):
			ax.text(x=j, y=i, s=confMat[i, j], ha="center", va="center")
	plt.xticks(range(10))
	plt.yticks(range(10))
	plt.xlabel("Predictions")
	plt.ylabel("Actual")
	title = "Training" if isTraining else "Test"
	plt.title(f"{title} Confusion Matrix\nAccuracy = {accuracy}")
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

with open("C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\mnist.txt", "r") as file:
	data = file.readlines()[1:] # Skip header

xTrain, yTrain, xTest, yTest = extractData(data)

clf = NeuralNetwork()

choice = input("Enter F to use file parameters or T to train network from scratch: ").upper()

if choice == "F":
	with open("biasesAndWeights.txt", "r") as file:
		params = file.read().split("\n---\n")

	hiddenBias = np.array(params[0].split("\n")).astype(float).reshape(-1, 1)
	hiddenWeights = np.array([line.split() for line in params[1].split("\n")]).astype(float)
	outputBias = np.array(params[2].split("\n")).astype(float).reshape(-1, 1)
	outputWeights = np.array([line.split() for line in params[3].split("\n")]).astype(float)

	clf.hiddenBias = hiddenBias
	clf.hiddenWeights = hiddenWeights
	clf.outputBias = outputBias
	clf.outputWeights = outputWeights
else:
	clf.fit(xTrain, yTrain)

	start = time.perf_counter()
	clf.train()
	end = time.perf_counter()

	s = ("\n".join(str(b) for b in clf.hiddenBias.flatten())
		+ "\n---\n"
		+ "\n".join((" ".join(str(i) for i in w)) for w in clf.hiddenWeights)
		+ "\n---\n"
		+ "\n".join(str(b) for b in clf.outputBias.flatten())
		+ "\n---\n"
		+ "\n".join((" ".join(str(i) for i in w)) for w in clf.outputWeights))
	with open("biasesAndWeights.txt", "w") as file:
		file.write(s)

	print(f"Done in {round(end - start, 3)}s. Saved biases and weights to file.")

# Plot confusion matrices

trainPredictions = [clf.predict(i) for i in xTrain]
testPredictions = [clf.predict(i) for i in xTest]
trainConfMat, trainAcc = confusionMatrix(trainPredictions, yTrain)
testConfMat, testAcc = confusionMatrix(testPredictions, yTest)

plotMatrix(True, trainConfMat, trainAcc)
plotMatrix(False, testConfMat, testAcc)

# User draws a digit to predict

pg.init()
pg.display.set_caption("Draw a digit!")
scene = pg.display.set_mode((DRAWING_SIZE, DRAWING_SIZE))
userCoords = []
drawing = True
leftButtonDown = False

while drawing:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			drawing = False
			pg.quit()
		elif event.type == pg.MOUSEBUTTONDOWN:
			if event.button == 1:
				leftButtonDown = True
				x, y = event.pos
				userCoords.append([x, y])
				scene.set_at((x, y), (255, 255, 255))
				pg.display.flip()
		elif event.type == pg.MOUSEMOTION:
			if leftButtonDown:
				x, y = event.pos
				userCoords.append([x, y])
				scene.set_at((x, y), (255, 255, 255))
				pg.display.flip()
		elif event.type == pg.MOUSEBUTTONUP:
			if event.button == 1:
				leftButtonDown = False

userCoords = np.array(userCoords) // (DRAWING_SIZE // 27) # Make coords range from 0-27
userCoords = np.unique(userCoords, axis=0) # Keep unique pairs only
drawnDigit = np.zeros((28, 28))
drawnDigit[userCoords[:,1], userCoords[:,0]] = 1
plt.figure(figsize=(4, 4))
plt.imshow(drawnDigit, cmap="gray")
plt.title("Drawn digit")
plt.show()

drawnDigit = drawnDigit.reshape(1, 784)[0].astype(int)
predVector = clf.predict(drawnDigit)
print("\nDrawn digit is: {}  ({} % sure)".format(np.argmax(predVector), round(100 * np.max(predVector), 3)))

# Plot loss graph

if choice != "F":
	xPlot = list(range(len(clf.loss)))
	yPlot = clf.loss
	plt.figure(figsize=(8, 6))
	plt.plot(xPlot, yPlot, color="red")
	plt.xlabel("Training iteration")
	plt.ylabel("Mean loss")
	plt.title("Loss")
	plt.show()
