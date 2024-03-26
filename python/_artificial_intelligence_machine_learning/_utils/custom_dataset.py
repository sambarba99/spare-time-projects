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
		if isinstance(self.x[index], torch.Tensor):
			ret_x = self.x[index]
		else:
			ret_x = torch.tensor(self.x[index], dtype=torch.float32)

		if self.y:
			ret_y = [
				yi[index] if isinstance(yi[index], torch.Tensor)
				else torch.tensor(yi[index], dtype=torch.float32)
				for yi in self.y
			]
			return ret_x, *ret_y
		else:
			return ret_x

	def __len__(self):
		return self.n_samples
