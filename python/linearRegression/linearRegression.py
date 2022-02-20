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
def extract_data(data, train_test_ratio=0.5):
	feature_names = data.pop(0).strip("\n").split(",")

	data = [row.strip("\n").split() for row in data]
	np.random.shuffle(data)
	data = np.array(data).astype(float)

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

feature_names, x_train, y_train, x_test, y_test, data = extract_data(data)

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
print(f"Highest (abs) correlation with y ({feature_names[-1]}): {max_corr}  (feature '{feature_names[idx_max_corr]}')")

weights = ", ".join(f"{we:.3f}" for we in regressor.weights)
x_plot = np.array(list(x_train[:, idx_max_corr]) + list(x_test[:, idx_max_corr]))
y_plot = regressor.weights[idx_max_corr] * x_plot + regressor.bias
y_scatter = list(y_train) + list(y_test)
plt.figure(figsize=(10, 8))
plt.scatter(x_plot, y_scatter, color="black", s=5)
plt.plot(x_plot, y_plot, color="red")
plt.xlabel(feature_names[idx_max_corr] + " (normalised)")
plt.ylabel(feature_names[-1] + " (normalised)")
plt.title(f"Gradient descent solution\nweights = {weights}\nbias = {regressor.bias:.3f}")
plt.show()

# Plot MSE graph

x_plot = list(range(len(regressor.cost_history)))
y_plot = np.array(regressor.cost_history) / len(x_train)
plt.figure(figsize=(8, 6))
plt.plot(x_plot, y_plot, color="red")
plt.xlabel("Training iteration")
plt.ylabel("Mean square error")
plt.title("MSE during training")
plt.show()
