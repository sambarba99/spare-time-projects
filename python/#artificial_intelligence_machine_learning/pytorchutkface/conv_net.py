"""
CNN class

Author: Sam Barba
Created 30/10/2022
"""

from torch import nn


class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv_block = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
			nn.ReLU(),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
			nn.ReLU(),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
			nn.ReLU(),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
			nn.ReLU(),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		self.age_branch = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features=30976, out_features=512),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=512),
			nn.Dropout(),  # 0.5
			nn.Linear(512, 1024),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(1024, 1)
		)
		self.gender_branch = nn.Sequential(
			nn.Flatten(),
			nn.Linear(30976, 256),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=256),
			nn.Dropout(),
			nn.Linear(256, 512),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(512, 1),
			nn.Sigmoid()
		)
		self.race_branch = nn.Sequential(
			nn.Flatten(),
			nn.Linear(30976, 256),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=256),
			nn.Dropout(),
			nn.Linear(256, 512),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(512, 5),  # 5 races
			nn.Softmax(dim=1)
		)


	def forward(self, x):
		conv_out = self.conv_block(x)
		# print(conv_out.shape)  # To get in_features of age/gender/race branches
		age_out = self.age_branch(conv_out)
		gender_out = self.gender_branch(conv_out)
		race_out = self.race_branch(conv_out)

		return age_out, gender_out, race_out
