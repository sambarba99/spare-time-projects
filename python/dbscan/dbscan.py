# DBSCAN demo
# Author: Sam Barba
# Created 27/12/2021

from dbscanclusterer import DBSCANclusterer
import matplotlib.pyplot as plt
import numpy as np

UNDEFINED = -1
NOISE = 0

# Change these
USE_FILE_DATA = True
NUM_POINTS = 500 # If not using file data

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if USE_FILE_DATA:
	with open("C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\dbscanData.txt", "r") as file:
		coords = file.readlines()[1:] # Skip header

	coords = [row.strip("\n").split() for row in coords]
	coords = np.array(coords).astype(float)
else:
	# Create random samples
	coords = np.random.uniform(-100, 100, size=(NUM_POINTS, 2))

# Initially all points are labelled as undefined
points = []
for c in coords:
	point = {"x": c[0], "y": c[1], "label": UNDEFINED}
	points.append(point)

# ---------------------------------- Cluster samples ---------------------------------- #

if USE_FILE_DATA:
	clusterer = DBSCANclusterer(epsilon=10, minPoints=12)
else:
	clusterer = DBSCANclusterer(epsilon=20, minPoints=17)

clusterer.predict(points)

# ---------------------------------- Plot clusters ---------------------------------- #

# Unique labels excluding noise
uniqueLabels = sorted(list(set(p["label"] for p in points if p["label"] != NOISE)))

# + 1 to include a 'cluster' for noise samples
clusters = [[] for _ in range(len(uniqueLabels) + 1)]

for p in points:
	# Any noise coords go into clusters[0], as noise label = 0
	coords = [p["x"], p["y"]]
	clusters[p["label"]].append(coords)

plt.figure(figsize=(8, 8))
for idx, c in enumerate(clusters):
	if idx == 0:
		if clusters[0]: # If there is noise...
			plt.scatter(*np.array(c).T, c="black") # ... Plot as black
	else:
		plt.scatter(*np.array(c).T, alpha=0.7)

legend = [f"Class {str(i)} ({len(clusters[i])} samples)" for i in uniqueLabels]
if clusters[0]: # If there is noise...
	legend.insert(0, f"Noise ({len(clusters[0])} samples)")

plt.legend(legend)
plt.show()
