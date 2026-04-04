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
			nn.Conv2d(1, 32, kernel_size=3, padding=1),    # -> (N, 32, 64, 64)
			nn.LeakyReLU(),
			nn.MaxPool2d(2),                               # -> (N, 32, 32, 32)
			nn.Conv2d(32, 128, kernel_size=3, padding=1),  # -> (N, 128, 32, 32)
			nn.LeakyReLU(),
			nn.MaxPool2d(3)                                # -> (N, 128, 10, 10)
		)
		self.fc_block = nn.Sequential(
			nn.Flatten(),                                  # -> (N, 12800)
			nn.Dropout(),  # 0.5
			nn.Linear(12800, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 1)
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		fc_out = self.fc_block(conv_out)

		return fc_out
