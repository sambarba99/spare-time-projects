"""
Custom dataset class

Author: Sam Barba
Created 26/03/2024
"""

import torch


class CustomDataset(torch.utils.data.Dataset):
	def __init__(self, x, *y):
		self.x = x
		self.y = y
		self.n_samples = len(x)

	def __getitem__(self, index):
		ret_x = self.x[index] \
			if isinstance(self.x[index], torch.Tensor) \
			else torch.FloatTensor(self.x[index])

		if self.y:
			ret_y = [
				yi[index] if isinstance(yi[index], torch.Tensor)
				else torch.FloatTensor(yi[index])
				for yi in self.y
			]
			return ret_x, *ret_y

		return ret_x

	def __len__(self):
		return self.n_samples
