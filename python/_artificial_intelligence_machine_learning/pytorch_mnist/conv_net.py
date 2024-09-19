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
			nn.Conv2d(1, 32, kernel_size=3),
			nn.LeakyReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3),
			nn.LeakyReLU(),
			nn.MaxPool2d(2)
		)
		self.fc_block = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(),  # 0.5
			nn.Linear(1600, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 10)  # 10 classes
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		fc_out = self.fc_block(conv_out).squeeze()

		return fc_out
