"""
Decision tree class

Author: Sam Barba
Created 01/10/2022
"""

import numpy as np
from sklearn.metrics import f1_score

class DecisionTree:
	def __init__(self, x, y, max_depth):
		self.left = None
		self.right = None
		self.is_leaf = None
		self.class_idx = None
		self.feature_idx = None
		self.split_threshold = None

		def find_best_split(x, y):
			"""
			Given a dataest and its target values, find the optimal combination of feature
			and split point that yields minimum gini impurity (or max info gain if using commented code)
			"""

			# def calculate_entropy(y):
			# 	if len(y) <= 1: return 0
			#
			# 	counts = y.bincount()
			# 	probs = counts[counts.nonzero()] / len(y)  # .nonzero ensures that we're not doing log(0) after
			#
			# 	return -(probs * np.log2(probs)).sum()

			def calculate_gini(y):
				if len(y) <= 1: return 0

				probs = np.bincount(y) / len(y)

				return 1 - (probs ** 2).sum()

			# parent_entropy = calculate_entropy(y)
			# best = {'info_gain': -1}
			best = {'gini_impurity': np.inf}

			# Loop every possible split of every dimension
			for i in range(x.shape[1]):
				for split_threshold in np.unique(x[:, i]):
					left_indices = np.where(x[:, i] <= split_threshold)
					right_indices = np.where(x[:, i] > split_threshold)
					left = y[left_indices]
					right = y[right_indices]

					# info_gain = parent_entropy - len(left) / len(y) * calculate_entropy(left) \
					# 	- len(right) / len(y) * calculate_entropy(right)

					# if info_gain > best['info_gain']:
					# 	best = {'feature_idx': i,
					# 		'split_threshold': split_threshold,
					# 		'info_gain': info_gain,
					# 		'left_indices': left_indices,
					# 		'right_indices': right_indices}

					gini_impurity = calculate_gini(left) * len(left) / len(y) \
						+ calculate_gini(right) * len(right) / len(y)
					if gini_impurity < best['gini_impurity']:
						best = {'feature_idx': i,
							'split_threshold': split_threshold,
							'gini_impurity': gini_impurity,
							'left_indices': left_indices,
							'right_indices': right_indices}

			return best

		# If depth is 0, or all remaining class labels are the same, this is a leaf node
		if max_depth == 0 or (y == y[0]).all():
			labels, counts = np.unique(y, return_counts=True)
			self.is_leaf = True
			self.class_idx = labels[counts.argmax()]
		else:
			split = find_best_split(x, y)
			self.left = DecisionTree(x[split['left_indices']], y[split['left_indices']], max_depth - 1)
			self.right = DecisionTree(x[split['right_indices']], y[split['right_indices']], max_depth - 1)
			self.is_leaf = False
			self.feature_idx = split['feature_idx']
			self.split_threshold = split['split_threshold']

	def predict(self, sample):
		if self.is_leaf:
			return self.class_idx
		if sample[self.feature_idx] <= self.split_threshold:
			return self.left.predict(sample)
		else:
			return self.right.predict(sample)

	def evaluate(self, x, y):
		predictions = np.array([self.predict(sample) for sample in x])
		n_classes = len(np.unique(y))

		return f1_score(y, predictions, average='binary' if n_classes == 2 else 'weighted')

	def get_depth(self):
		if self.is_leaf:
			return 0
		return max(self.left.get_depth(), self.right.get_depth()) + 1
