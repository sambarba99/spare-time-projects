"""
Early Stopping class for PyTorch models

Author: Sam Barba
Created 26/03/2024
"""

from copy import deepcopy

import numpy as np


class EarlyStopping:
	def __init__(self, *, model, patience, mode, track_best_weights, min_delta=0, print_precision_on_stop=4):
		"""
		Args:
			model (torch.nn.Module):
				The model being trained. Used to optionally save and restore the best-performing weights.
			patience (int):
				Number of consecutive epochs with no improvement before stopping.
			mode (str):
				'min' or 'max'. If 'min', training stops when monitored metric stops decreasing (e.g. loss).
				If 'max', training stops when monitored metric stops increasing (e.g. accuracy).
			track_best_weights (bool):
				Whether to keep track of the model's best-performing weights, so they can be restored later.
			min_delta (float):
				Minimum change in the monitored metric to qualify as an improvement.
			print_precision_on_stop (int):
				Number of decimal places to use when printing the best score upon early stopping.
				If None, no message is printed.
		"""

		assert patience >= 1
		assert mode in ('min', 'max')
		assert track_best_weights in (True, False)
		assert min_delta >= 0
		assert print_precision_on_stop is None or print_precision_on_stop >= 0

		self.model = model
		self.patience = patience
		self.track_best_weights = track_best_weights
		self.min_delta = min_delta
		self.print_precision_on_stop = print_precision_on_stop
		self.epoch = 0
		self.num_bad_epochs = 0
		self.best_weights = deepcopy(model.state_dict()) if track_best_weights else None
		self.best_score = np.inf if mode == 'min' else -np.inf
		self.monitor_op = np.less if mode == 'min' else np.greater
		self.multiplier = -1 if mode == 'min' else 1

	def __call__(self, new_score):
		self.epoch += 1

		if self.monitor_op(new_score, self.best_score + self.multiplier * self.min_delta):
			self.num_bad_epochs = 0
			self.best_score = new_score
			if self.track_best_weights:
				self.best_weights = deepcopy(self.model.state_dict())
		else:
			self.num_bad_epochs += 1

			if self.num_bad_epochs >= self.patience:
				if self.print_precision_on_stop is not None:
					print(f'Early stopping (best score = {self.best_score:.{self.print_precision_on_stop}f},'
						f' epoch {self.epoch - self.patience})')

				return True

		return False

	def restore_best_weights(self):
		assert self.track_best_weights, 'track_best_weights needs to be True in order to restore them.'
		self.model.load_state_dict(self.best_weights)
