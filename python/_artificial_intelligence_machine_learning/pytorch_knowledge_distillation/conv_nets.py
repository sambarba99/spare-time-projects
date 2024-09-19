"""
Teacher/student CNN classes

Author: Sam Barba
Created 29/09/2024
"""

from torch import nn


class Teacher(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv_block = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.MaxPool2d(2)
		)
		self.fc_block = nn.Sequential(
			nn.Flatten(),
			nn.Linear(8192, 2048),
			nn.LeakyReLU(),
			nn.Dropout(),  # 0.5
			nn.Linear(2048, 128),
			nn.LeakyReLU(),
			nn.Dropout(),
			nn.Linear(128, 10)  # 10 classes in dataset
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		fc_out = self.fc_block(conv_out).squeeze()

		return fc_out


class Student(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv_block = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.MaxPool2d(2)
		)
		self.fc_block = nn.Sequential(
			nn.Flatten(),
			nn.Linear(8192, 256),
			nn.LeakyReLU(),
			nn.Dropout(),
			nn.Linear(256, 10)
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		fc_out = self.fc_block(conv_out).squeeze()

		return fc_out
