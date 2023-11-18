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
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
			nn.LeakyReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
			nn.LeakyReLU(),
			nn.MaxPool2d(kernel_size=2)
		)

		self.fc_block = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(),  # 0.5
			nn.Linear(in_features=1600, out_features=128),
			nn.LeakyReLU(),
			nn.Linear(in_features=128, out_features=10),  # 10 classes
			nn.Softmax(dim=-1)
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		# print(conv_out.shape)  # To get in_features of self.fc_block (e.g. [N, 64, 17, 14] -> 64x17x14 = 15232)
		fc_out = self.fc_block(conv_out)
		return fc_out
