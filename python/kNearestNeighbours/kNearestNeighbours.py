# KNN demo
# Author: Sam Barba
# Created 11/09/2021

from knnclassifier import KNN
import matplotlib.pyplot as plt
import numpy as np
import random

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def confusionMatrix(predictions, actual):
	numClasses = len(np.unique(actual))
	confMat = np.zeros((numClasses, numClasses)).astype(int)

	for a, p in zip(actual, predictions):
		confMat[a, p] += 1

	accuracy = np.trace(confMat) / confMat.sum()
	return confMat, accuracy

def plotMatrix(k, confMat, accuracy):
	fig, ax = plt.subplots(figsize=(6, 7))
	ax.matshow(confMat, cmap=plt.cm.Blues, alpha=0.7)
	ax.xaxis.set_ticks_position("bottom")
	for i in range(confMat.shape[0]):
		for j in range(confMat.shape[1]):
			ax.text(x=j, y=i, s=confMat[i, j], ha="center", va="center")
	plt.xlabel("Predictions")
	plt.ylabel("Actual")
	plt.title(f"Confusion Matrix (optimal k = {k})\nAccuracy = {accuracy}")
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

choice = input("Enter I to use iris dataset or W for wine dataset: ").upper()
print()

if choice == "I":
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\irisData.txt"
else:
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\wineData.txt"

with open(path, "r") as file:
	data = file.readlines()[1:] # Skip header

data = [row.strip("\n").split() for row in data]
random.shuffle(data)
data = np.array(data).astype(float)
x, y = data[:,:-1], data[:,-1].astype(int)

bestAcc = bestK = -1
bestConfMat = None

for k in range(3, int(len(data) ** 0.5), 2):
	clf = KNN(k)
	clf.fit(x, y)

	predictions = [clf.predict(i) for i in x]
	confMat, acc = confusionMatrix(predictions, y)

	if acc > bestAcc:
		bestAcc = acc
		bestK = k
		bestConfMat = confMat

	print(f"Accuracy with k = {k}: {acc}")

plotMatrix(bestK, bestConfMat, bestAcc)
