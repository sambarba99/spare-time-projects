"""
Decision tree class

Author: Sam Barba
Created 01/10/2022
"""

import numpy as np
from sklearn.metrics import f1_score


class DecisionTree:
	def __init__(self, x, y, max_depth, use_gini=True):
		self.left = None
		self.right = None
		self.is_leaf = None
		self.class_probs = None
		self.class_idx = None
		self.feature_idx = None
		self.split_threshold = None


		def find_best_split(x, y):
			"""
			Given a dataset and its target values, find the optimal combination of feature
			and split point that yields minimum Gini impurity (or max info gain)
			"""

			def calculate_gini(y):
				if len(y) <= 1: return 0

				probs = np.bincount(y) / len(y)

				return 1 - (probs * probs).sum()

			def calculate_entropy(y):
				if len(y) <= 1: return 0

				counts = np.bincount(y)
				probs = counts[counts.nonzero()] / len(y)  # nonzero() ensures that we're not doing log(0) after

				return -(probs * np.log2(probs)).sum()


			if use_gini:
				parent_entropy = None
				best = {'gini_impurity': np.inf}
			else:
				parent_entropy = calculate_entropy(y)
				best = {'info_gain': -1}

			# Loop every possible split of every dimension
			for i in range(x.shape[1]):
				for split_threshold in np.unique(x[:, i]):
					left_indices = x[:, i] <= split_threshold
					right_indices = ~left_indices
					left_y = y[left_indices]
					right_y = y[right_indices]

					if use_gini:
						left_gini_impurity = calculate_gini(left_y) * len(left_y) / len(y)
						right_gini_impurity = calculate_gini(right_y) * len(right_y) / len(y)
						gini_impurity = left_gini_impurity + right_gini_impurity

						if gini_impurity < best['gini_impurity']:
							best = {'feature_idx': i,
								'split_threshold': split_threshold,
								'gini_impurity': gini_impurity,
								'left_indices': left_indices,
								'right_indices': right_indices}
					else:
						left_entropy = calculate_entropy(left_y) * len(left_y) / len(y)
						right_entropy = calculate_entropy(right_y) * len(right_y) / len(y)
						info_gain = parent_entropy - left_entropy - right_entropy

						if info_gain > best['info_gain']:
							best = {'feature_idx': i,
								'split_threshold': split_threshold,
								'info_gain': info_gain,
								'left_indices': left_indices,
								'right_indices': right_indices}

			return best

		# If depth is 0, or all remaining class labels are the same, this is a leaf node
		one_class_left = (y == y[0]).all()
		if max_depth == 0 or one_class_left:
			labels, counts = np.unique(y, return_counts=True)
			if one_class_left:
				# Make class_probs[:, 1] the probability of the positive class (1)
				self.class_probs = [1, 0] if y[0] == 0 else [0, 1]
			else:
				self.class_probs = counts / len(y)
			self.class_idx = labels[counts.argmax()]
			self.is_leaf = True
		else:
			split = find_best_split(x, y)
			self.left = DecisionTree(x[split['left_indices']], y[split['left_indices']], max_depth - 1, use_gini)
			self.right = DecisionTree(x[split['right_indices']], y[split['right_indices']], max_depth - 1, use_gini)
			self.is_leaf = False
			self.feature_idx = split['feature_idx']
			self.split_threshold = split['split_threshold']

	def predict(self, x):
		if self.is_leaf:
			return {'class': self.class_idx, 'class_probs': self.class_probs}
		if x[self.feature_idx] <= self.split_threshold:
			return self.left.predict(x)
		else:
			return self.right.predict(x)

	def evaluate(self, x, y):
		pred_classes = [self.predict(xi)['class'] for xi in x]

		return f1_score(y, pred_classes, average='binary' if len(np.unique(y)) == 2 else 'weighted')

	@property
	def depth(self):
		if self.is_leaf:
			return 0
		return max(self.left.depth, self.right.depth) + 1
