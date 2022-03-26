# Linear regression demo
# Author: Sam Barba
# Created 10/11/2021

from linearregressor import LinearRegressor
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Split file data into train/test
def extract_data(path, train_test_ratio=0.5):
	data = np.genfromtxt(path, dtype=str, delimiter="\n")
	feature_names = data[0].strip("\n").split(",")
	# Skip header and convert to floats
	data = [row.split() for row in data[1:]]
	data = np.array(data).astype(float)
	np.random.shuffle(data)

	# Normalise data (x and y) (column-wise)
	data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

	x, y = data[:,:-1], data[:,-1]

	split = int(len(data) * train_test_ratio)

	x_train, y_train = x[:split], y[:split]
	x_test, y_test = x[split:], y[split:]

	return feature_names, x_train, y_train, x_test, y_test, data

def analytic_solution(x, y):
	# Adding dummy x0 = 1 makes the first weight w0 equal the bias
	x = [[1] + list(i) for i in list(x)]
	x = np.array(x)
	solution = ((np.linalg.inv(x.T.dot(x))).dot(x.T)).dot(y)
	weights, bias = solution[1:], solution[0]
	return weights, bias

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	choice = input("Enter B to use Boston housing dataset,"
		+ "\nC for car value dataset,"
		+ "\nor M for medical insurance dataset: ").upper()

	if choice == "B":
		path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\bostonData.txt"
	elif choice == "C":
		path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\carValueData.txt"
	else:
		path = "C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\medicalInsuranceData.txt"

	feature_names, x_train, y_train, x_test, y_test, data = extract_data(path)

	weights, bias = analytic_solution(x_train, y_train)
	weights = ", ".join(f"{we:.3f}" for we in weights)
	print(f"\nAnalytic solution:\n weights = {weights}\n bias = {bias:.3f}\n")

	regressor = LinearRegressor()
	regressor.fit(x_train, y_train)
	regressor.train()

	print("Training MSE:", regressor.cost_history[-1] / len(x_train))
	print("Test MSE:", regressor.cost(x_test, y_test, regressor.weights, regressor.bias) / len(x_test))

	# Plot regression line using column with the strongest correlation with y variable

	corr_coeffs = np.corrcoef(data.T)
	# Make bottom-right coefficient 0, as this doesn't count (correlation of last column with itself)
	corr_coeffs[-1,-1] = 0

	# Index of column that has the strongest correlation with y
	idx_max_corr = np.argmax(np.abs(corr_coeffs[:, -1]))
	max_corr = corr_coeffs[idx_max_corr, -1]

	print("\nFeature names:", ", ".join(feature_names))
	print(f"Highest (abs) correlation with y ({feature_names[-1]}): {max_corr} "
		f"(feature '{feature_names[idx_max_corr]}')")

	weights = ", ".join(f"{we:.3f}" for we in regressor.weights)
	x_plot = np.append(x_train[:, idx_max_corr], x_test[:, idx_max_corr])
	y_plot = regressor.weights[idx_max_corr] * x_plot + regressor.bias
	y_scatter = np.append(y_train, y_test)
	plt.figure(figsize=(10, 8))
	plt.scatter(x_plot, y_scatter, color="black", alpha=0.6, s=10)
	plt.plot(x_plot, y_plot, color="red")
	plt.xlabel(feature_names[idx_max_corr] + " (normalised)")
	plt.ylabel(feature_names[-1] + " (normalised)")
	plt.title(f"Gradient descent solution\nweights = {weights}\nbias = {regressor.bias:.3f}")
	plt.show()

	# Plot MSE graph

	y_plot = np.array(regressor.cost_history) / len(x_train)
	plt.figure(figsize=(8, 6))
	plt.plot(y_plot, color="red")
	plt.xlabel("Training iteration")
	plt.ylabel("Mean square error")
	plt.title("MSE during training")
	plt.show()

if __name__ == "__main__":
	main()
