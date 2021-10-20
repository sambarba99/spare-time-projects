# PCA demo
# Author: Sam Barba
# Created 10/11/2021

import matplotlib.pyplot as plt
import numpy as np
from PCA import PCA

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
	data = file.readlines()[1:] # Skip header

data = [row.replace("\n", "").split() for row in data]
data = np.array(data).astype(float)

x, y = data[:,:-1], data[:,-1].astype(int)

pca = PCA()
xTransform, newVariability = pca.transform(x)

plt.figure(figsize=(8, 8))
for classLabel in np.unique(y):
	plt.scatter(*xTransform[y == classLabel].T)
plt.legend(classes)
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.title(f"Shape of x: {x.shape}\nShape of PCA transform: {xTransform.shape}\nCaptured variability: {round(100 * newVariability, 2)}%")
plt.show()
