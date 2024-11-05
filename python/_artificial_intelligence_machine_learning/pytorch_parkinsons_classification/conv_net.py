"""
CNN class

Author: Sam Barba
Created 27/01/2024
"""

from torch import nn


class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv_block = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding=1),   # -> (N, 32, 64, 64)
			nn.LeakyReLU(),
			nn.MaxPool2d(2),                              # -> (N, 32, 32, 32)
			nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (N, 64, 32, 32)
			nn.LeakyReLU(),
			nn.MaxPool2d(3)                               # -> (N, 64, 10, 10)
		)
		self.fc_block = nn.Sequential(
			nn.Flatten(),                                 # -> (N, 6400)
			nn.Dropout(),  # 0.5
			nn.Linear(6400, 2048),
			nn.LeakyReLU(),
			nn.Linear(2048, 1)
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		fc_out = self.fc_block(conv_out).squeeze()

		return fc_out
