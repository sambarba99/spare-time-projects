# K-Means clustering demo
# Author: Sam Barba
# Created 21/11/2021

from KMeans import KMeans
import numpy as np

NUM_CLUSTERS = 4

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def makeRandomSamples(numClusters, numSamplesPerCluster=300):
	clusterCentroidCoords = np.random.uniform(-100, 100, size=(numClusters, 2))

	samples = []
	for c in clusterCentroidCoords:
		for _ in range(numSamplesPerCluster):
			# Generate random points radially around centroid x,y
			theta = np.random.uniform(0, 2 * np.pi)
			r = np.random.uniform(0, 50)
			x = r * np.cos(theta) + c[0]
			y = r * np.sin(theta) + c[1]
			samples.append([x, y])

	return np.array(samples)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

x = makeRandomSamples(NUM_CLUSTERS)

kMeans = KMeans(NUM_CLUSTERS)
yPred = kMeans.predict(x, True)
