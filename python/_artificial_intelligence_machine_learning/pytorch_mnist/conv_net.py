"""
CNN class

Author: Sam Barba
Created 30/10/2022
"""

from torch import nn


class CNN(nn.Module):
	def __init__(self):
		super().__init__()
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
			nn.Linear(400, 64),
			nn.LeakyReLU(),
			nn.Linear(64, 10)  # 10 classes
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		fc_out = self.fc_block(conv_out)

		return fc_out
