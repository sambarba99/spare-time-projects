"""
Dataset class

Author: Sam Barba
Created 06/03/2024
"""

import torch


class CustomDataset(torch.utils.data.Dataset):
	def __init__(self, x, y_age, y_gender, y_race):
		self.x = x
		self.y_age = torch.from_numpy(y_age).float()
		self.y_gender = torch.from_numpy(y_gender).float()
		self.y_race = torch.from_numpy(y_race).float()

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y_age[idx], self.y_gender[idx], self.y_race[idx]
