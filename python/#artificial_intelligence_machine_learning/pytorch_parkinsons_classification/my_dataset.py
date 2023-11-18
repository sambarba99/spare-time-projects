"""
Dataset class

Author: Sam Barba
Created 27/01/2024
"""

import torch


class MyDataset(torch.utils.data.Dataset):
	def __init__(self, x, y):
		self.x = torch.from_numpy(x).float()
		self.y = torch.from_numpy(y).float()
		self.n_samples = x.shape[0]

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def __len__(self):
		return self.n_samples
