# K-Means clustering demo
# Author: Sam Barba
# Created 21/11/2021

from KMeans import KMeans
import numpy as np

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def makeRandomSamples(numClusters, numSamplesPerCluster=300):
	clusterCentroidCoords = np.random.uniform(-100, 100, size=(numClusters, 2))

	clusterSamples = []
	for c in clusterCentroidCoords:
		for i in range(numSamplesPerCluster):
			# Generate random points radially around cluster x,y
			theta = np.random.uniform(0, 2 * np.pi)
			r = np.random.uniform(0, 50)
			sampleX = r * np.cos(theta) + c[0]
			sampleY = r * np.sin(theta) + c[1]
			clusterSamples.append([sampleX, sampleY])

	return np.array(clusterSamples)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

numClusters = int(input("Enter no. clusters: "))

x = makeRandomSamples(numClusters)

kMeans = KMeans(numClusters)
yPred = kMeans.predict(x, True)
