# SVM demo
# Author: Sam Barba
# Created 22/11/2021

import matplotlib.pyplot as plt
import numpy as np
from svm import SVM

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Split file data into train/test
def extract_data(data, train_test_ratio=0.5):
	data = [row.strip("\n").split() for row in data]
	np.random.shuffle(data)
	data = np.array(data).astype(float)

	x, y = data[:,:-1], data[:,-1].astype(int)

	split = int(len(data) * train_test_ratio)

	x_train, y_train = x[:split], y[:split]
	x_test, y_test = x[split:], y[split:]

	return x_train, y_train, x_test, y_test

def confusion_matrix(predictions, actual):
	predictions[predictions == -1] = 0
	actual[actual == -1] = 0

	num_classes = len(np.unique(actual))
	conf_mat = np.zeros((num_classes, num_classes)).astype(int)

	for a, p in zip(actual, predictions):
		conf_mat[a, p] += 1

	accuracy = np.trace(conf_mat) / conf_mat.sum()
	return conf_mat, accuracy

def plot_matrix(is_training, conf_mat, accuracy):
	fig, ax = plt.subplots(figsize=(6, 7))
	ax.matshow(conf_mat, cmap=plt.cm.Blues, alpha=0.7)
	ax.xaxis.set_ticks_position("bottom")
	for i in range(conf_mat.shape[0]):
		for j in range(conf_mat.shape[1]):
			ax.text(x=j, y=i, s=conf_mat[i, j], ha="center", va="center")
	plt.xlabel("Predictions")
	plt.ylabel("Actual")
	title = "Training" if is_training else "Test"
	plt.title(f"{title} Confusion Matrix\nAccuracy = {accuracy}")
	plt.show()

def get_hyperplane_value(x, weights, bias, offset):
	return (-weights[0] * x + bias + offset) / weights[1]

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

with open("C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\svmData.txt", "r") as file:
	data = file.readlines()[1:] # Skip header

x_train, y_train, x_test, y_test = extract_data(data)

clf = SVM()
clf.fit(x_train, y_train)
clf.train()

# Plot confusion matrices

train_predictions = clf.predict(x_train)
test_predictions = clf.predict(x_test)
train_conf_mat, train_acc = confusion_matrix(train_predictions, y_train)
test_conf_mat, test_acc = confusion_matrix(test_predictions, y_test)

plot_matrix(True, train_conf_mat, train_acc)
plot_matrix(False, test_conf_mat, test_acc)

# Visualise SVM

x_scatter = np.array(list(x_train) + list(x_test))
y_scatter = np.array(list(y_train) + list(y_test))

plt.figure(figsize=(8, 8))
for class_label in np.unique(y_scatter):
	plt.scatter(*x_scatter[y_scatter == class_label].T, alpha=0.7)
plt.legend(["class -1", "class 1"])

hyperplane_start_x = np.min(x_scatter[:, 0])
hyperplane_end_x = np.max(x_scatter[:, 0])
hyperplane_start_y = get_hyperplane_value(hyperplane_start_x, clf.weights, clf.bias, 0)
hyperplane_end_y = get_hyperplane_value(hyperplane_end_x, clf.weights, clf.bias, 0)
negative_plane_start_y = get_hyperplane_value(hyperplane_start_x, clf.weights, clf.bias, -1)
negative_plane_end_y = get_hyperplane_value(hyperplane_end_x, clf.weights, clf.bias, -1)
positive_plane_start_y = get_hyperplane_value(hyperplane_start_x, clf.weights, clf.bias, 1)
positive_plane_end_y = get_hyperplane_value(hyperplane_end_x, clf.weights, clf.bias, 1)

plt.plot([hyperplane_start_x, hyperplane_end_x], [hyperplane_start_y, hyperplane_end_y], color="black", ls="--")
plt.plot([hyperplane_start_x, hyperplane_end_x], [negative_plane_start_y, negative_plane_end_y], color="red")
plt.plot([hyperplane_start_x, hyperplane_end_x], [positive_plane_start_y, positive_plane_end_y], color="red")

y_min = np.min(x_scatter[:, 1])
y_max = np.max(x_scatter[:, 1])
plt.ylim([y_min - 0.5, y_max + 0.5])

w = ", ".join(f"{we:.3f}" for we in clf.weights)
m = -clf.weights[0] / clf.weights[1]
c = -clf.bias / clf.weights[1]

plt.title(f"Weights: {w}\nBias: {clf.bias:.3f}\nm: {m:.3f}  |  c: {c:.3f}")
plt.show()
