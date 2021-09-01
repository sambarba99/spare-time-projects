# KNN demo
# Author: Sam Barba
# Created 11/09/2021

import random

# ---------------------------------------------------------------------------------------------------- #
# ---------------------------------------------  CLASSES  -------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

class KNN:
	def __init__(self, k):
		self.k = k

	# Training set (x) and corresponding class labels (y)
	def fit(self, xTrain, yTrain):
		self.xTrain = xTrain
		self.yTrain = yTrain

	def predictClasses(self, xTest):
		return [self.__predictClass(testItem) for testItem in xTest]

	def __predictClass(self, testItem):
		# Compute distance between test sample and all training samples
		distances = [self.__euclideanDist(testItem, trainItem) for trainItem in self.xTrain]

		# Index each calculated distance, i.e.: [(0, 1.24914), (1, 0.4812)...]
		idxAndDistances = list(enumerate(distances))

		# Sort in ascending order by distance, but keep only first k training samples (nearest)
		nearestK = sorted(idxAndDistances, key = lambda tup: tup[1])[:k]

		# Get classifications of nearest k samples
		classesOfNearestK = [self.yTrain[tup[0]] for tup in nearestK]

		# Return mode of these
		return max(set(classesOfNearestK), key = classesOfNearestK.count)

	def __euclideanDist(self, testItem, trainItem):
		return sum((float(attr1) - float(attr2)) ** 2 for attr1, attr2 in zip(testItem, trainItem)) ** 0.5

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def convertStringColToInt(data, col):
	strVals = [row[col] for row in data]
	uniqueStrVals = set(strVals)
	lookup = dict(enumerate(uniqueStrVals))
	lookup = {v: k for k, v in lookup.items()} # Dictionary inversion

	for row in data:
		row[col] = lookup[row[col]]

def evaluateCrossValidation(data, numFolds, k):
	folds = crossValidationSplit(data, numFolds)
	foldScores = []
	clf = KNN(k)

	for fold in folds:
		# Test set (x) and corresponding class labels (y) (i.e. last column)
		xTest = [row[:-1] for row in fold]
		yTest = [row[-1] for row in fold]

		# Training set (x) and class labels (y)
		xTrain = [[row[:-1] for row in f] for f in folds if f != fold]
		yTrain = [[row[-1] for row in f] for f in folds if f != fold]

		# 'Flatten' xTrain and yTrain
		xTrain = sum(xTrain, [])
		yTrain = sum(yTrain, [])

		clf.fit(xTrain, yTrain)
		predictions = clf.predictClasses(xTest)

		foldScores.append(getAccuracy(predictions, yTest))

	return foldScores

# Split data into n folds
def crossValidationSplit(data, numFolds):
	random.shuffle(data)
	foldSize = len(data) // numFolds
	folds = [data[i:i + foldSize] for i in range(0, len(data), foldSize)]

	return folds

def getAccuracy(predicted, actual):
	matches = [1 if p == a else 0 for p, a in zip(predicted, actual)]
	return sum(matches) / len(actual)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

while True:
	choice = input("Enter: I to use iris data,"
		+ "\n D for diabetes data,"
		+ "\n or T for Titanic survivor data: ").upper()[0]

	if choice == "I": file = open("irisData.csv", "r")
	elif choice == "D": file = open("diabetesData.csv", "r")
	else: file = open("titanicSurvivorData.csv", "r")

	data = file.readlines()[1:] # Skip header
	file.close()
	
	data = [row.replace("\n","").split(",") for row in data]

	if choice == "T":
		# Convert passenger sex column from male/female to 0/1
		convertStringColToInt(data, 1)

	numFolds = k = int(input("\nInput no. folds (= k): "))

	trainingTestRatio = 1 - 1 / numFolds

	print("\nTraining set to test set size ratio:", round(trainingTestRatio, 2))

	foldScores = evaluateCrossValidation(data, numFolds, k)
	print("\nFold scores:", foldScores)
	print("\nMean accuracy: {}%\n".format(round(100 * sum(foldScores) / len(foldScores), 2)))

	if choice == "I": randData = [5.2, 2.8, 3.5, 1.1]
	elif choice == "D": randData = [4, 180, 80, 40, 500, 38, 1.5, 35]
	else: randData = [1, 1, 30, 2, 1, 200]

	clf = KNN(k)
	xTrain = [row[:-1] for row in data]
	yTrain = [row[-1] for row in data]
	clf.fit(xTrain, yTrain)
	predictedClass = clf.predictClasses([randData])[0]

	print("Random data: {}    Predicted class (using all data): {}".format(str(randData), predictedClass))

	choice = input("\nEnter to continue or X to exit: ").upper()
	if len(choice) > 0 and choice[0] == 'X':
		break
	print()
