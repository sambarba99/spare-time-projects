"""
CNN class

Author: Sam Barba
Created 09/04/2026
"""

from torch import nn


class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		# Input shape (N, 1, 256, 256) (batch size, no. colour channels, height, width)
		self.conv_block = nn.Sequential(
			nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=1),  # -> (N, 8, 127, 127)
			nn.LeakyReLU(),
			nn.MaxPool2d(2),                                      # -> (N, 8, 63, 63)
			nn.Conv2d(8, 16, kernel_size=3, padding=1),           # -> (N, 16, 63, 63)
			nn.LeakyReLU(),
			nn.MaxPool2d(2),                                      # -> (N, 16, 31, 31)
			nn.Conv2d(16, 32, kernel_size=3, padding=1),          # -> (N, 32, 31, 31)
			nn.LeakyReLU(),
			nn.MaxPool2d(3)                                       # -> (N, 32, 10, 10)
		)
		self.fc_block = nn.Sequential(
			nn.Flatten(),                                         # -> (N, 3200)
			nn.Dropout(),  # 0.5
			nn.Linear(3200, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 4)  # 4 classes in dataset
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		fc_out = self.fc_block(conv_out)

		return fc_out
