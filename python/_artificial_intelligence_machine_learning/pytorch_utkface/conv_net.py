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
			nn.Conv2d(3, 64, kernel_size=3, padding=1),     # -> (N, 64, 128, 128)
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),    # -> (N, 64, 128, 128)
			nn.ReLU(),
			nn.MaxPool2d(2),                                # -> (N, 64, 64, 64)
			nn.Conv2d(64, 128, kernel_size=3, padding=1),   # -> (N, 128, 64, 64)
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),  # -> (N, 128, 64, 64)
			nn.ReLU(),
			nn.MaxPool2d(2),                                # -> (N, 128, 32, 32)
			nn.Conv2d(128, 256, kernel_size=3, padding=1),  # -> (N, 256, 32, 32)
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),  # -> (N, 256, 32, 32)
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),  # -> (N, 256, 32, 32)
			nn.ReLU(),
			nn.MaxPool2d(2),                                # -> (N, 256, 16, 16)
			nn.Conv2d(256, 512, kernel_size=3, padding=1),  # -> (N, 512, 16, 16)
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),  # -> (N, 512, 16, 16)
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),  # -> (N, 512, 16, 16)
			nn.ReLU(),
			nn.MaxPool2d(2),                                # -> (N, 512, 8, 8)
			nn.Conv2d(512, 512, kernel_size=3, padding=1),  # -> (N, 512, 8, 8)
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),  # -> (N, 512, 8, 8)
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),  # -> (N, 512, 8, 8)
			nn.ReLU(),
			nn.MaxPool2d(2)                                 # -> (N, 512, 4, 4)
		)
		self.fc_block = nn.Sequential(
			nn.Flatten(),                                   # -> (N, 8192)
			nn.Dropout(),  # 0.5
			nn.Linear(8192, 2048),
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
			nn.Linear(128, 1)
		)
		self.race_branch = nn.Sequential(
			nn.Linear(2048, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 5)  # 5 race categories in dataset
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		fc_out = self.fc_block(conv_out)
		age_out = self.age_branch(fc_out)
		gender_out = self.gender_branch(fc_out)
		race_out = self.race_branch(fc_out)

		return age_out, gender_out, race_out
