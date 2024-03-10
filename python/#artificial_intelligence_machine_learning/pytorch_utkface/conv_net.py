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
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		self.fc_block = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(),  # 0.5
			nn.Linear(in_features=8192, out_features=2048),
			nn.LeakyReLU()
		)
		self.age_branch = nn.Sequential(
			nn.Linear(2048, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 1)
		)
		self.gender_branch = nn.Sequential(
			nn.Linear(2048, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 1),
			nn.Sigmoid()
		)
		self.race_branch = nn.Sequential(
			nn.Linear(2048, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 5),  # 5 race categories in dataset
			nn.Softmax(dim=-1)
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		# print(conv_out.shape)  # To get in_features of self.fc_block (e.g. [N, 64, 17, 14] -> 64x17x14 = 15232)
		fc_out = self.fc_block(conv_out)
		age_out = self.age_branch(fc_out)
		gender_out = self.gender_branch(fc_out)
		race_out = self.race_branch(fc_out)

		return age_out, gender_out, race_out
