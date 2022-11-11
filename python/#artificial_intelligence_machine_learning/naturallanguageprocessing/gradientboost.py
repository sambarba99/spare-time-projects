"""
Gradient boost classifier for NLP demo

Author: Sam Barba
Created 29/09/2022
"""

from numpy import exp, log, where
from sklearn.tree import DecisionTreeRegressor

class GradientBoost:
	def __init__(self, *, learning_rate, n_trees, max_depth):
		self.initial_log_odd = 0
		self.trees = []
		self.leaf_keys = []
		self.n_trees = n_trees
		self.learning_rate = learning_rate
		self.max_depth = max_depth

	def fit(self, x_train, y_train):
		y_train = where(y_train == -1, 0, 1)

		# Calculate log odds
		pos_odds = (y_train == 1).sum() / (y_train == 0).sum()
		log_odds = log(pos_odds)
		self.initial_log_odd = log_odds

		# Convert log odds to probilities
		pos_prob = exp(log_odds) / (1 + exp(log_odds))

		# Set probabilities to initial probability for all data points
		probabilities = [pos_prob] * len(y_train)
		log_probabilities = [log_odds] * len(y_train)

		# Calculate the residual for every data point
		residuals = [observation - pos_prob for observation in y_train]

		# Loop through trees
		self.trees = []
		self.leaf_keys = []
		for tree_idx in range(self.n_trees):
			print(f'Tree {tree_idx + 1}/{self.n_trees}', end=' ')

			# Make new tree
			self.trees.append(DecisionTreeRegressor(max_depth=self.max_depth))

			# Fit new tree to the residual
			self.trees[tree_idx].fit(x_train, residuals)
			predictions = self.trees[tree_idx].predict(x_train)

			# Assign each datapoint to leaf
			leaves, leaf_vals = {}, {}
			for idx, pred in enumerate(predictions):
				leaf_list = leaves.get(pred, [])
				leaf_list.append(idx)
				leaves[pred] = leaf_list

			# Get leaf value for each leaf
			for leaf_idx in list(leaves.keys()):
				numerator = denominator = 0
				for entry_idx in leaves[leaf_idx]:
					numerator += residuals[entry_idx]
					denominator += (probabilities[entry_idx] * (1 - probabilities[entry_idx]))

				leaf_value = numerator / denominator
				leaf_vals[leaf_idx] = leaf_value

			self.leaf_keys.append(leaf_vals)

			# For each data point
			for entry_idx in range(len(y_train)):
				# Get leaf value for the data point
				leaf_update = leaf_vals[predictions[entry_idx]]

				# Update log odds with leaf value (multiplied by the learning rate)
				entry_log_odds = log_probabilities[entry_idx] + leaf_update * self.learning_rate

				# Calculate new probability for data point
				entry_prob = exp(entry_log_odds) / (1 + exp(entry_log_odds))

				# Calculate the data point residual
				entry_residual = y_train[entry_idx] - entry_prob

				# Update the data point log odds
				log_probabilities[entry_idx] = entry_log_odds

				# Update the data point probability
				probabilities[entry_idx] = entry_prob

				# Update the data point residual
				residuals[entry_idx] = entry_residual

	def predict(self, x):
		# Set all probabilities to the initial log odds
		log_odds = [self.initial_log_odd] * len(x)
		n_predictions = len(x)

		# Loop through trees to update log odds
		for idx, tree in enumerate(self.trees):
			# Get prediction class for each data point
			predictions = tree.predict(x)

			# Update the log odds based on the data points prediction from the tree
			log_odds = [log_odds[i] + (self.learning_rate * self.leaf_keys[idx][predictions[i]]) for i in range(n_predictions)]

		# Convert log odds to probabilities
		probs = [exp(log_odd) / (1 + exp(log_odd)) for log_odd in log_odds]
		class_predictions = where(probs > 0.5, 1, -1)
		return class_predictions
