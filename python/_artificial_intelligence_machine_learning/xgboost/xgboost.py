"""
XGBoost class

Author: Sam Barba
Created 09/03/2024
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class XGBoostClassifier:
	def __init__(self, *, n_estimators, max_depth, learning_rate, n_classes):
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.learning_rate = learning_rate
		self.n_classes = n_classes
		self.models = []

	def __softmax(self, x):
		exps = np.exp(x - x.max(axis=1, keepdims=True))
		return exps / np.sum(exps, axis=1, keepdims=True)

	def __calculate_gradient(self, y_true, y_pred):
		return y_true - self.__softmax(y_pred)

	def fit(self, x, y):
		# Initialise the predictions with zeros
		pred = np.zeros((len(y), self.n_classes))

		# One-hot encode the target variable
		y_one_hot = np.eye(self.n_classes)[y]

		for _ in range(self.n_estimators):
			# Calculate the gradient
			gradient = self.__calculate_gradient(y_one_hot, pred)

			# Fit a weak learner (decision tree) to the gradient, then store it
			tree = DecisionTreeRegressor(max_depth=self.max_depth)
			tree.fit(x, gradient)
			self.models.append(tree)

			# Update the predictions using the new weak learner
			pred += self.learning_rate * tree.predict(x)

	def predict(self, x):
		# Initialise predictions with zeros
		pred = np.zeros((x.shape[0], self.n_classes))

		# Make predictions using each weak learner
		for model in self.models:
			pred += self.learning_rate * model.predict(x)

		# Apply softmax function to convert to probabilities
		probs = self.__softmax(pred)
		return probs
