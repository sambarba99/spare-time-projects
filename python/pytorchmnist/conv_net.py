"""
CNN class

Author: Sam Barba
Created 30/10/2022
"""

from torch import nn

class CNN(nn.Module):
	def __init__(self, n_classes):
		super(CNN, self).__init__()
		self.conv_block = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2)),
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2))
		)
		self.fc_block = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(),  # 0.5
			nn.Linear(in_features=1600, out_features=n_classes),
			nn.Softmax(dim=1)
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		# print(conv_out.shape)  # To determine in_features of fc_block
		fc_out = self.fc_block(conv_out)
		return fc_out
