"""
Custom dataset class

Author: Sam Barba
Created 26/03/2024
"""

import torch


class CustomDataset(torch.utils.data.Dataset):
	def __init__(self, x, *y):
		assert isinstance(x, torch.Tensor) or all(isinstance(xi, torch.Tensor) for xi in x)
		assert all(isinstance(yi, torch.Tensor) for yi in y)

		self.x = x
		self.y = y
		self.n_samples = len(x)

	def __getitem__(self, index):
		if self.y:
			ret_y = [yi[index] for yi in self.y]
			return self.x[index], *ret_y

		return self.x[index]

	def __len__(self):
		return self.n_samples
