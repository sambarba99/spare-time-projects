# Naive Bayes classification demo
# Author: Sam Barba
# Created 21/11/2021

from naivebayesclassifier import NaiveBayesClassifier
import matplotlib.pyplot as plt
import numpy as np
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

choice = input("Enter B to use breast tumour dataset,"
	+ "\nI for iris dataset,"
	+ "\nP for pulsar dataset,"
	+ "\nT for Titanic dataset,"
	+ "\nor W for wine dataset: ").upper()

if choice == "B":
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\breastTumourData.txt"
elif choice == "I":
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\irisData.txt"
elif choice == "P":
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\pulsarData.txt"
elif choice == "T":
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\titanicData.txt"
else:
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\wineData.txt"

with open(path, "r") as file:
	data = file.readlines()[1:] # Skip header

xTrain, yTrain, xTest, yTest = extractData(data)

clf = NaiveBayesClassifier()
clf.fit(xTrain, yTrain)
clf.train()

# Plot confusion matrices

trainPredictions = [clf.predict(i) for i in xTrain]
testPredictions = [clf.predict(i) for i in xTest]
trainConfMat, trainAcc = confusionMatrix(trainPredictions, yTrain)
testConfMat, testAcc = confusionMatrix(testPredictions, yTest)

plotMatrix(True, trainConfMat, trainAcc)
plotMatrix(False, testConfMat, testAcc)
