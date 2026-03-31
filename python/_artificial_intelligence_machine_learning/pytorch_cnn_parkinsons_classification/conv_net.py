"""
CNN class

Author: Sam Barba
Created 27/01/2024
"""

from torch import nn


class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		# Input shape (N, 1, 64, 64) (batch size, no. colour channels, height, width)
		self.conv_block = nn.Sequential(
			nn.Conv2d(1, 8, kernel_size=3, padding=1),   # -> (N, 8, 64, 64)
			nn.LeakyReLU(),
			nn.MaxPool2d(2),                             # -> (N, 8, 32, 32)
			nn.Conv2d(8, 16, kernel_size=3, padding=1),  # -> (N, 16, 32, 32)
			nn.LeakyReLU(),
			nn.MaxPool2d(3)                              # -> (N, 16, 10, 10)
		)
		self.fc_block = nn.Sequential(
			nn.Flatten(),                                # -> (N, 1600)
			nn.Dropout(),  # 0.5
			nn.Linear(1600, 512),
			nn.LeakyReLU(),
			nn.Linear(512, 1)
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		fc_out = self.fc_block(conv_out)

		return fc_out
