"""
Early Stopping class

Author: Sam Barba
Created 26/03/2024
"""

import numpy as np


class EarlyStopping:
	def __init__(self, *, patience, min_delta, mode):
		assert mode in ('min', 'max')

		self.patience = patience
		self.min_delta = min_delta
		self.trigger_count = 0
		self.best_weights = None
		self.best_score = np.inf if mode == 'min' else -np.inf
		self.monitor_op = np.less if mode == 'min' else np.greater
		self.multiplier = -1 if mode == 'min' else 1

	def __call__(self, new_score, model_weights):
		if self.monitor_op(new_score, self.best_score + self.multiplier * self.min_delta):
			self.trigger_count = 0
			self.best_score = new_score
			self.best_weights = model_weights
		else:
			self.trigger_count += 1

		return self.trigger_count >= self.patience
