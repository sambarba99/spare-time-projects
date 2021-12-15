# Linear regression demo
# Author: Sam Barba
# Created 10/11/2021

from LinearRegressor import LinearRegressor
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

	# Normalise data (x and y) (column-wise)
	data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

	x, y = data[:,:-1], data[:,-1]

	split = int(len(data) * trainTestRatio)

	xTrain, yTrain = x[:split], y[:split]
	xTest, yTest = x[split:], y[split:]

	return featureNames, xTrain, yTrain, xTest, yTest, data

def analyticSolution(x, y):
	# Adding dummy x0 = 1 makes the first weight w0 equal the bias
	x = [[1] + list(i) for i in list(x)]
	x = np.array(x)
	solution = ((np.linalg.inv(x.T.dot(x))).dot(x.T)).dot(y)
	weights, bias = solution[1:], solution[0]
	return weights, bias

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

choice = input("Enter B to use Boston housing dataset,"
	+ "\nC for car value dataset,"
	+ "\nor M for medical insurance dataset: ").upper()

if choice == "B":
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\bostonData.txt"
elif choice == "C":
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\carValueData.txt"
else:
	path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\medicalInsuranceData.txt"

with open(path, "r") as file:
	data = file.readlines()

featureNames, xTrain, yTrain, xTest, yTest, data = extractData(data)

weights, bias = analyticSolution(xTrain, yTrain)
bias = round(bias, 3)
weights = ", ".join(str(round(i, 3)) for i in weights)
print(f"\nAnalytic solution:\n weights = {weights}\n bias = {bias}\n")

regressor = LinearRegressor()
regressor.fit(xTrain, yTrain)
regressor.train()

print("Training MSE:", regressor.costHistory[-1] / len(xTrain))
print("Test MSE:", regressor.cost(xTest, yTest, regressor.weights, regressor.bias) / len(xTest))

# Plot regression line using column with the strongest correlation with y variable

corrCoeffs = np.corrcoef(data.T)
# Make bottom-right coefficient 0, as this doesn't count (correlation of last column with itself)
corrCoeffs[-1,-1] = 0

# Index of column that has the strongest correlation with y
idxMaxCorr = np.argmax(np.abs(corrCoeffs[:,-1]))
maxCorr = corrCoeffs[idxMaxCorr, -1]

print("\nFeature names:", ", ".join(featureNames))
print(f"Highest (abs) correlation with y ({featureNames[-1]}): {maxCorr}  (feature '{featureNames[idxMaxCorr]}')")

weights = ", ".join(str(round(i, 3)) for i in regressor.weights)
xPlot = np.array(list(xTrain[:, idxMaxCorr]) + list(xTest[:, idxMaxCorr]))
yPlot = regressor.weights[idxMaxCorr] * xPlot + regressor.bias
yScatter = list(yTrain) + list(yTest)
plt.figure(figsize=(10, 8))
plt.scatter(xPlot, yScatter, color="black", s=5)
plt.plot(xPlot, yPlot, color="red")
plt.xlabel(featureNames[idxMaxCorr] + " (normalised)")
plt.ylabel(featureNames[-1] + " (normalised)")
plt.title("Gradient descent solution\nweights = {}\nbias = {}".format(weights, round(regressor.bias, 3)))
plt.show()

# Plot MSE graph

xPlot = list(range(len(regressor.costHistory)))
yPlot = np.array(regressor.costHistory) / len(xTrain)
plt.figure(figsize=(8, 6))
plt.plot(xPlot, yPlot, color="red")
plt.xlabel("Training iteration")
plt.ylabel("Mean square error")
plt.title("MSE during training")
plt.show()
