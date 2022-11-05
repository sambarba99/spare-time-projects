"""
Early Stopping class

Author: Sam Barba
Created 30/10/2022
"""

import torch

class EarlyStopping:
	def __init__(self, *, patience, min_delta):
		self.patience = patience
		self.min_delta = min_delta
		self.trigger_times = 0
		self.last_loss_val = torch.inf
		self.best_loss_val = torch.inf
		self.best_weights = None

	def check_stop(self, new_loss_val, model_weights):
		if self.last_loss_val - new_loss_val < self.min_delta:
			self.trigger_times += 1
			if self.trigger_times >= self.patience:
				return True
		else:
			self.trigger_times = 0
			if new_loss_val < self.best_loss_val:  # Keep track of best model
				self.best_loss_val = new_loss_val
				self.best_weights = model_weights

		self.last_loss_val = new_loss_val
