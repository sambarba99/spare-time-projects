"""
Decision tree class

Author: Sam Barba
Created 19/10/2022
"""

import numpy as np
from sklearn.metrics import mean_squared_error


class DecisionTree:
	def __init__(self, x, y, max_depth):
		self.left = None
		self.right = None
		self.is_leaf = None
		self.value = None
		self.feature_idx = None
		self.split_threshold = None


		def find_best_split(x, y):
			"""
			Given a dataset and its target values, find the optimal combination of
			feature and split point that yields minimum MSE
			"""

			def calculate_mse(y):
				if y.shape[0] <= 1:
					return 0
				pred = np.full(y.shape[0], y.mean())  # Use mean as the prediction

				return mean_squared_error(y, pred)


			best = {'mse': np.inf}

			# Loop every possible split of every dimension
			for i in range(x.shape[1]):
				for split_threshold in np.unique(x[:, i]):
					left_indices = x[:, i] <= split_threshold
					right_indices = ~left_indices
					left = y[left_indices]
					right = y[right_indices]

					left_mse = calculate_mse(left) * len(left) / len(y)
					right_mse = calculate_mse(right) * len(right) / len(y)
					mse = left_mse + right_mse

					if mse < best['mse']:
						best = {'feature_idx': i,
							'split_threshold': split_threshold,
							'mse': mse,
							'left_indices': left_indices,
							'right_indices': right_indices}

			return best

		# If depth is 0, or only 1 value of y left, this is a leaf node
		if max_depth == 0 or len(y) <= 1:
			self.is_leaf = True
			self.value = y.mean() if len(y) else 0
		else:
			split = find_best_split(x, y)
			self.left = DecisionTree(x[split['left_indices']], y[split['left_indices']], max_depth - 1)
			self.right = DecisionTree(x[split['right_indices']], y[split['right_indices']], max_depth - 1)
			self.is_leaf = False
			self.feature_idx = split['feature_idx']
			self.split_threshold = split['split_threshold']

	def predict(self, sample):
		if self.is_leaf:
			return self.value
		if sample[self.feature_idx] <= self.split_threshold:
			return self.left.predict(sample)
		else:
			return self.right.predict(sample)

	def evaluate(self, x, y):
		y_pred = np.array([self.predict(sample) for sample in x])

		return mean_squared_error(y, y_pred) ** 0.5  # RMSE

	@property
	def depth(self):
		if self.is_leaf:
			return 0
		return max(self.left.depth, self.right.depth) + 1
