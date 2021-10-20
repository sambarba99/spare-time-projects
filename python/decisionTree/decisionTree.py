# Decision tree demo
# Author: Sam Barba
# Created 03/11/2021

import matplotlib.pyplot as plt
import numpy as np
import random

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Split file data into train/test
def extractData(data, trainTestRatio=0.5):
	featureNames = data.pop(0).replace("\n", "").split(",")

	data = [row.replace("\n", "").split() for row in data]
	random.shuffle(data)
	data = np.array(data).astype(float)

	x, y = data[:,:-1], data[:,-1].astype(int)

	split = int(len(data) * trainTestRatio)

	xTrain, yTrain = x[:split], y[:split]
	xTest, yTest = x[split:], y[split:]

	return featureNames, xTrain, yTrain, xTest, yTest

def buildTree(x, y, maxDepth=1000):
	# Generate leaf node if either stopping condition has been reached
	if maxDepth == 1 or np.all(y == y[0]):
		classes, counts = np.unique(y, return_counts=True)
		return {"leaf": True, "class": classes[np.argmax(counts)]}
	else:
		move = findBestSplit(x, y)
		left = buildTree(x[move["leftIndices"]], y[move["leftIndices"]], maxDepth - 1)
		right = buildTree(x[move["rightIndices"]], y[move["rightIndices"]], maxDepth - 1)

		return {"leaf": False,
			"feature": move["feature"],
			"splitThreshold": move["splitThreshold"],
			"infogain": move["infogain"],
			"left": left,
			"right": right}

# Given a dataset and its target values, find the optimal combination 
# of feature and split point that yields maximum information gain
def findBestSplit(x, y):
	parentEntropy = calculateEntropy(y)
	best = {"infogain": -1}

	# Loop every possible split of every dimension
	for i in range(x.shape[1]):
		for splitThreshold in np.unique(x[:,i]):
			leftIndices = np.where(x[:,i] <= splitThreshold)
			rightIndices = np.where(x[:,i] > splitThreshold)
			left = y[leftIndices]
			right = y[rightIndices]
			infogain = parentEntropy - len(left) / len(y) * calculateEntropy(left) - len(right) / len(y) * calculateEntropy(right)

			if infogain > best["infogain"]:
				best = {"feature": i,
					"splitThreshold": splitThreshold,
					"infogain": infogain,
					"leftIndices": leftIndices,
					"rightIndices": rightIndices}

	return best

def calculateEntropy(y):
	if len(y) <= 1: return 0

	counts = np.bincount(y)
	probs = counts[np.nonzero(counts)] / len(y) # np.nonzero ensures that we're not doing log(0) after

	return -(probs * np.log2(probs)).sum()

# Test different max depth values and create tree with best one
def makeBestTree(xTrain, yTrain, xTest, yTest):
	bestTree = None
	bestDepth = bestTrainAcc = bestTestAcc = -1
	depth = 2

	while True:
		tree, trainAcc, testAcc = evaluate(xTrain, yTrain, xTest, yTest, depth)
		print(f"Depth {depth}: training accuracy = {trainAcc} | test accuracy = {testAcc}")

		conditions = [trainAcc >= bestTrainAcc,
			testAcc >= bestTestAcc,
			not (trainAcc == bestTrainAcc and testAcc == bestTestAcc)]

		if all(conditions):
			bestTree = tree
			bestDepth = depth
			bestTrainAcc = trainAcc
			bestTestAcc = testAcc
		else:
			break # No improvement, so stop

		depth += 1

	return bestTree, bestDepth

def evaluate(xTrain, yTrain, xTest, yTest, maxDepth):
	tree = buildTree(xTrain, yTrain, maxDepth)
	trainPredictions = np.array([predict(tree, sample) for sample in xTrain])
	testPredictions = np.array([predict(tree, sample) for sample in xTest])
	trainAcc = (trainPredictions == yTrain).sum() / len(yTrain)
	testAcc = (testPredictions == yTest).sum() / len(yTest)

	return tree, trainAcc, testAcc

def predict(tree, sample):
	if tree["leaf"]:
		return tree["class"]
	else:
		if sample[tree["feature"]] <= tree["splitThreshold"]:
			return predict(tree["left"], sample)
		else:
			return predict(tree["right"], sample)

def printTree(tree, classes, indent=0):
	global featureNames

	if tree["leaf"]:
		print(" " * indent + classes[tree["class"]])
	else:
		f = tree["feature"]
		print("{}x{} ({}) <= {}".format(" " * indent, f, featureNames[f], tree["splitThreshold"]))
		printTree(tree["left"], classes, indent + 4)
		print("{}x{} ({}) > {}".format(" " * indent, f, featureNames[f], tree["splitThreshold"]))
		printTree(tree["right"], classes, indent + 4)

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
print()

if choice == "B":
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\breastTumourData.txt"
	classes = ["malignant", "benign"]
elif choice == "I":
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\irisData.txt"
	classes = ["setosa", "versicolor", "virginica"]
elif choice == "P":
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\pulsarData.txt"
	classes = ["not pulsar", "pulsar"]
elif choice == "T":
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\titanicData.txt"
	classes = ["did not survive", "survived"]
else:
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\wineData.txt"
	classes = ["class 0", "class 1", "class 2"]

with open(path, "r") as file:
	data = file.readlines()

featureNames, xTrain, yTrain, xTest, yTest = extractData(data)

tree, depth = makeBestTree(xTrain, yTrain, xTest, yTest)

print(f"\nOptimal tree (depth {depth}):\n")

printTree(tree, classes)

# Plot confusion matrices

trainPredictions = [predict(tree, i) for i in xTrain]
testPredictions = [predict(tree, i) for i in xTest]
trainConfMat, trainAcc = confusionMatrix(trainPredictions, yTrain)
testConfMat, testAcc = confusionMatrix(testPredictions, yTest)

plotMatrix(True, trainConfMat, trainAcc)
plotMatrix(False, testConfMat, testAcc)
