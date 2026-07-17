"""
Early Stopping class

Author: Sam Barba
Created 2024-03-26
"""

from copy import deepcopy

import numpy as np
import torch


class EarlyStopping:
	def __init__(self, *, target, patience, mode, track_best_weights=True, min_delta=0):
		"""
		Args:
			target (torch.nn.Module or torch.nn.Parameter):
				The model or parameter being trained. Used to optionally save and restore the best-performing weights.
			patience (int):
				Number of consecutive epochs with no improvement before stopping.
			mode (str):
				'min' or 'max'. If 'min', training stops when monitored metric stops decreasing (e.g. loss).
				If 'max', training stops when monitored metric stops increasing (e.g. accuracy).
			track_best_weights (bool):
				Whether to keep track of the target's best-performing weights, so they can be restored later.
			min_delta (float):
				Minimum change in the monitored metric to qualify as an improvement.
		"""

		assert isinstance(target, (torch.nn.Module, torch.nn.Parameter)), 'target to track must be nn.Module or nn.Parameter'
		assert isinstance(patience, int) and patience >= 1
		assert mode in ('min', 'max')
		assert track_best_weights in (True, False)
		assert isinstance(min_delta, (float, int)) and min_delta >= 0

		self.target = target
		self.is_module = isinstance(target, torch.nn.Module)
		self.patience = patience
		self.track_best_weights = track_best_weights
		self.min_delta = min_delta
		self.epoch = 0
		self.num_bad_epochs = 0
		if track_best_weights:
			self.best = deepcopy(target.state_dict()) if self.is_module else target.detach().clone()
		else:
			self.best = None
		self.best_score = np.inf if mode == 'min' else -np.inf
		self.monitor_op = np.less if mode == 'min' else np.greater
		self.multiplier = -1 if mode == 'min' else 1

	def __call__(self, new_score):
		self.epoch += 1

		if self.monitor_op(new_score, self.best_score + self.multiplier * self.min_delta):
			self.num_bad_epochs = 0
			self.best_score = new_score
			if self.track_best_weights:
				self.best = deepcopy(self.target.state_dict()) if self.is_module else self.target.detach().clone()
		else:
			self.num_bad_epochs += 1
			if self.num_bad_epochs >= self.patience:
				return True

		return False

	def print_stop_message(self, precision=4):
		print(f'Early stopping (best score = {self.best_score:.{precision}f}, epoch {self.epoch - self.patience})')

	def restore_best_weights(self):
		assert self.track_best_weights, 'track_best_weights needs to be True in order to restore them.'

		if self.is_module:
			self.target.load_state_dict(self.best)
		else:
			with torch.no_grad():
				self.target.copy_(self.best)
