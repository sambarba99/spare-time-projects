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
NUM_POINTS = 500  # If not using file data

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if USE_FILE_DATA:
	coords = np.genfromtxt("C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\dbscanData.txt", dtype=str, delimiter="\n")
	# Skip header and convert to floats
	coords = [row.split() for row in coords[1:]]
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
	clusterer = DBSCANclusterer(epsilon=10, min_points=12)
else:
	clusterer = DBSCANclusterer(epsilon=20, min_points=17)

clusterer.predict(points)

# ---------------------------------- Plot clusters ---------------------------------- #

# Unique labels excluding noise
unique_labels = sorted(list(set(p["label"] for p in points if p["label"] != NOISE)))

# + 1 to include a 'cluster' for noise samples
clusters = [[] for _ in range(len(unique_labels) + 1)]

for p in points:
	# Any noise coords go into clusters[0], as noise label = 0
	coords = [p["x"], p["y"]]
	clusters[p["label"]].append(coords)

plt.figure(figsize=(8, 8))
for idx, c in enumerate(clusters):
	if idx == 0 and clusters[0]:
		# If there's noise, plot it as black
		plt.scatter(*np.array(c).T, c="black", label=f"Noise ({len(c)} samples)")
	else:
		plt.scatter(*np.array(c).T, alpha=0.7, label=f"Cluster {idx} ({len(c)} samples)")

plt.legend()
plt.show()
