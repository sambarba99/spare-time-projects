# Logistic regression demo
# Author: Sam Barba
# Created 10/11/2021

from logisticregressor import LogisticRegressor
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (8, 6)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Split file data into train/test
def extract_data(path, train_test_ratio=0.5):
	data = np.genfromtxt(path, dtype=str, delimiter="\n")
	feature_names = data[0].strip().split(",")
	# Skip header and convert to floats
	data = [row.split() for row in data[1:]]
	data = np.array(data).astype(float)
	np.random.shuffle(data)

	x, y = data[:,:-1], data[:,-1].astype(int)

	# Normalise data (column-wise) (no need for y)
	x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

	split = int(len(data) * train_test_ratio)

	x_train, y_train = x[:split], y[:split]
	x_test, y_test = x[split:], y[split:]

	return feature_names, x_train, y_train, x_test, y_test, data

def confusion_matrix(predictions, actual):
	num_classes = len(np.unique(actual))
	conf_mat = np.zeros((num_classes, num_classes)).astype(int)

	for a, p in zip(actual, predictions):
		conf_mat[a, p] += 1

	accuracy = np.trace(conf_mat) / conf_mat.sum()
	return conf_mat, accuracy

def plot_matrix(is_training, conf_mat, accuracy):
	ax = plt.subplot()
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

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
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

	feature_names, x_train, y_train, x_test, y_test, data = extract_data(path)

	regressor = LogisticRegressor()
	regressor.fit(x_train, y_train)
	regressor.train()

	# Plot confusion matrices

	train_conf_mat, train_acc = confusion_matrix(regressor.predict(x_train), y_train)
	test_conf_mat, test_acc = confusion_matrix(regressor.predict(x_test), y_test)

	plot_matrix(True, train_conf_mat, train_acc)
	plot_matrix(False, test_conf_mat, test_acc)

	# Plot regression line using 2 columns with the strongest correlation with y (class)

	corr_coeffs = np.corrcoef(data.T)
	# Make bottom-right coefficient 0, as this doesn't count (correlation of last column with itself)
	corr_coeffs[-1,-1] = 0

	# Indices of columns in descending order of correlation with y
	indices = np.argsort(np.abs(corr_coeffs[:, -1]))[::-1]
	# Keep 2 strongest
	indices = indices[:2]
	max_corr = corr_coeffs[indices, -1]

	print(f"\nHighest (abs) correlation with y (class): {max_corr[0]}  (feature '{feature_names[indices[0]]}')")
	print(f"2nd highest (abs) correlation with y (class): {max_corr[1]}  (feature '{feature_names[indices[1]]}')")

	w1, w2 = regressor.weights[indices]
	m = -w1 / w2
	c = -regressor.bias / w2
	x_scatter = np.append(x_train[:, indices], x_test[:, indices], axis=0)
	y_scatter = np.append(y_train, y_test)
	x_plot = np.array([np.min(x_scatter, axis=0)[0], np.max(x_scatter, axis=0)[0]])
	y_plot = m * x_plot + c

	plt.figure()
	for idx, class_label in enumerate(np.unique(y_scatter)):
		plt.scatter(*x_scatter[y_scatter == class_label].T, alpha=0.7, label=classes[idx])
	plt.plot(x_plot, y_plot, color="black", ls="--")
	plt.ylim(np.min(x_scatter) * 1.1, np.max(x_scatter) * 1.1)
	plt.xlabel(feature_names[indices[0]] + " (normalised)")
	plt.ylabel(feature_names[indices[1]] + " (normalised)")
	plt.title(f"Gradient descent solution\nm = {m:.3f}  |  c = {c:.3f}")
	plt.legend()
	plt.show()

	# Plot cost graph

	plt.figure()
	plt.plot(regressor.cost_history, color="red")
	plt.xlabel("Training iteration")
	plt.ylabel("Cost")
	plt.title("Cost during training")
	plt.show()

if __name__ == "__main__":
	main()
