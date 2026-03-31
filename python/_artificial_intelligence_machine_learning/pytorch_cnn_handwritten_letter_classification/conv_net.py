"""
CNN class

Author: Sam Barba
Created 28/11/2024
"""

from torch import nn


class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		# Input shape (N, 1, 28, 28) (batch size, no. colour channels, height, width)
		self.conv_block = nn.Sequential(
			nn.Conv2d(1, 8, kernel_size=3),   # -> (N, 8, 26, 26)
			nn.LeakyReLU(),
			nn.MaxPool2d(2),                  # -> (N, 8, 13, 13)
			nn.Conv2d(8, 16, kernel_size=3),  # -> (N, 16, 11, 11)
			nn.LeakyReLU(),
			nn.MaxPool2d(2)                   # -> (N, 16, 5, 5)
		)
		self.fc_block = nn.Sequential(
			nn.Flatten(),                     # -> (N, 400)
			nn.Dropout(),  # 0.5
			nn.Linear(400, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 26)  # 26 classes
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		fc_out = self.fc_block(conv_out)

		return fc_out
