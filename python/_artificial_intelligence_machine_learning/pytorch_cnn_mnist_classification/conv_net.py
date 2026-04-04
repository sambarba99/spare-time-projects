"""
CNN class

Author: Sam Barba
Created 30/10/2022
"""

from torch import nn


class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		# Input shape (N, 1, 28, 28) (batch size, no. colour channels, height, width)
		self.conv_block = nn.Sequential(
			nn.Conv2d(1, 8, kernel_size=3, padding=1),   # -> (N, 8, 28, 28)
			nn.LeakyReLU(),
			nn.MaxPool2d(2),                             # -> (N, 8, 14, 14)
			nn.Conv2d(8, 32, kernel_size=3, padding=1),  # -> (N, 32, 14, 14)
			nn.LeakyReLU(),
			nn.MaxPool2d(2)                              # -> (N, 32, 7, 7)
		)
		self.fc_block = nn.Sequential(
			nn.Flatten(),                                # -> (N, 1568)
			nn.Dropout(),  # 0.5
			nn.Linear(1568, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 10)  # 10 classes (0-9)
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		fc_out = self.fc_block(conv_out)

		return fc_out
