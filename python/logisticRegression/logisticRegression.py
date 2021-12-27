# Logistic regression demo
# Author: Sam Barba
# Created 10/11/2021

from LogisticRegressor import LogisticRegressor
import matplotlib.pyplot as plt
import numpy as np
import random

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Split file data into train/test
def extractData(data, trainTestRatio=0.5):
	featureNames = data.pop(0).strip("\n").split(",")

	data = [row.strip("\n").split() for row in data]
	random.shuffle(data)
	data = np.array(data).astype(float)

	x, y = data[:,:-1], data[:,-1].astype(int)

	# Normalise data (column-wise) (no need for y)
	x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

	split = int(len(data) * trainTestRatio)

	xTrain, yTrain = x[:split], y[:split]
	xTest, yTest = x[split:], y[split:]

	return featureNames, xTrain, yTrain, xTest, yTest, data

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
	+ "\nP for pulsar dataset,"
	+ "\nor T for Titanic dataset: ").upper()

if choice == "B":
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\breastTumourData.txt"
	classes = ["malignant", "benign"]
elif choice == "P":
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\pulsarData.txt"
	classes = ["not pulsar", "pulsar"]
else:
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\titanicData.txt"
	classes = ["did not survive", "survived"]

with open(path, "r") as file:
	data = file.readlines()

featureNames, xTrain, yTrain, xTest, yTest, data = extractData(data)

regressor = LogisticRegressor()
regressor.fit(xTrain, yTrain)
regressor.train()

# Plot confusion matrices

trainConfMat, trainAcc = confusionMatrix(regressor.predict(xTrain), yTrain)
testConfMat, testAcc = confusionMatrix(regressor.predict(xTest), yTest)

plotMatrix(True, trainConfMat, trainAcc)
plotMatrix(False, testConfMat, testAcc)

# Plot regression line using 2 columns with the strongest correlation with y (class)

corrCoeffs = np.corrcoef(data.T)
# Make bottom-right coefficient 0, as this doesn't count (correlation of last column with itself)
corrCoeffs[-1,-1] = 0

# Indices of columns in descending order of correlation with y
indices = np.argsort(np.abs(corrCoeffs[:,-1]))[::-1]
# Keep 2 strongest
indices = indices[:2]
maxCorr = corrCoeffs[indices, -1]

print(f"\nHighest (abs) correlation with y (class): {maxCorr[0]}  (feature '{featureNames[indices[0]]}')")
print(f"2nd highest (abs) correlation with y (class): {maxCorr[1]}  (feature '{featureNames[indices[1]]}')")

w1, w2 = regressor.weights[indices]
m = -w1 / w2
c = -regressor.bias / w2
xScatter = np.array(list(xTrain[:, indices]) + list(xTest[:, indices]))
yScatter = np.array(list(yTrain) + list(yTest))
xPlot = np.array([np.min(xScatter, axis=0)[0], np.max(xScatter, axis=0)[0]])
yPlot = m * xPlot + c
plt.figure(figsize=(8, 8))
for classLabel in np.unique(yScatter):
	plt.scatter(*xScatter[yScatter == classLabel].T, alpha=0.7)
plt.legend(classes)
plt.plot(xPlot, yPlot, color="black", ls="--")
plt.ylim(np.min(xScatter) * 1.1, np.max(xScatter) * 1.1)
plt.xlabel(featureNames[indices[0]] + " (normalised)")
plt.ylabel(featureNames[indices[1]] + " (normalised)")
plt.title(f"Gradient descent solution\nm = {m:.3f}  |  c = {c:.3f}")
plt.show()

# Plot cost graph

xPlot = list(range(len(regressor.costHistory)))
yPlot = regressor.costHistory
plt.figure(figsize=(8, 6))
plt.plot(xPlot, yPlot, color="red")
plt.xlabel("Training iteration")
plt.ylabel("Cost")
plt.title("Cost during training")
plt.show()
