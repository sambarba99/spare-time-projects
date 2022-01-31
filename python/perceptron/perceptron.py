# Perceptron demo
# Author: Sam Barba
# Created 23/11/2021

import matplotlib.pyplot as plt
import numpy as np
from perceptronclassifier import PerceptronClf
import random

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Split file data into train/test
def extractData(data, trainTestRatio=0.5):
	data = [row.strip("\n").split() for row in data]
	random.shuffle(data)
	data = np.array(data).astype(float)

	x, y = data[:,:-1], data[:,-1].astype(int)
	# File data is for SVM testing, so convert class -1 to 0
	y[y == -1] = 0

	split = int(len(data) * trainTestRatio)

	xTrain, yTrain = x[:split], y[:split]
	xTest, yTest = x[split:], y[split:]

	return xTrain, yTrain, xTest, yTest

def confusionMatrix(predictions, actual):
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
	plt.xlabel("Predictions")
	plt.ylabel("Actual")
	title = "Training" if isTraining else "Test"
	plt.title(f"{title} Confusion Matrix\nAccuracy = {accuracy}")
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Use SVM testing data
with open("C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\svmData.txt", "r") as file:
	data = file.readlines()[1:] # Skip header

xTrain, yTrain, xTest, yTest = extractData(data)

clf = PerceptronClf()
clf.fit(xTrain, yTrain)
clf.train()

# Plot confusion matrices

trainPredictions = clf.predict(xTrain)
testPredictions = clf.predict(xTest)
trainConfMat, trainAcc = confusionMatrix(trainPredictions, yTrain)
testConfMat, testAcc = confusionMatrix(testPredictions, yTest)

plotMatrix(True, trainConfMat, trainAcc)
plotMatrix(False, testConfMat, testAcc)

# Visualise perceptron

xScatter = np.array(list(xTrain) + list(xTest))
yScatter = np.array(list(yTrain) + list(yTest))

plt.figure(figsize=(8, 8))
for classLabel in np.unique(yScatter):
	plt.scatter(*xScatter[yScatter == classLabel].T, alpha=0.7)
plt.legend(["class 0", "class 1"])

decisionBoundX1 = np.min(xScatter[:,0])
decisionBoundX2 = np.max(xScatter[:,0])
decisionBoundY1 = (-clf.weights[0] * decisionBoundX1 - clf.bias) / clf.weights[1]
decisionBoundY2 = (-clf.weights[0] * decisionBoundX2 - clf.bias) / clf.weights[1]

plt.plot([decisionBoundX1, decisionBoundX2], [decisionBoundY1, decisionBoundY2], color="black", ls="--")

yMin = np.min(xScatter[:,1])
yMax = np.max(xScatter[:,1])
plt.ylim([yMin - 0.5, yMax + 0.5])

w = ", ".join(f"{we:.3f}" for we in clf.weights)
b = round(clf.bias, 3)
m = round(-clf.weights[0] / clf.weights[1], 3)
c = round(-clf.bias / clf.weights[1], 3)

plt.title(f"Weights: {w}\nBias: {b}\nm: {m} | c: {c}")
plt.show()
