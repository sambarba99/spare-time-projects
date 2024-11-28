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
			nn.Conv2d(3, 16, kernel_size=3, padding=1),   # -> (N, 16, 32, 32)
			nn.LeakyReLU(),
			nn.Conv2d(16, 16, kernel_size=3, padding=1),  # -> (N, 16, 32, 32)
			nn.LeakyReLU(),
			nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> (N, 32, 32, 32)
			nn.LeakyReLU(),
			nn.MaxPool2d(2),                              # -> (N, 32, 16, 16)
			nn.Conv2d(32, 32, kernel_size=3, padding=1),  # -> (N, 32, 16, 16)
			nn.LeakyReLU(),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (N, 64, 16, 16)
			nn.LeakyReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),  # -> (N, 64, 16, 16)
			nn.LeakyReLU(),
			nn.MaxPool2d(2)                               # -> (N, 64, 8, 8)
		)
		self.fc_block = nn.Sequential(
			nn.Flatten(),                                 # -> (N, 4096)
			nn.Dropout(),  # 0.5
			nn.Linear(4096, 1024),
			nn.LeakyReLU(),
			nn.Dropout(),
			nn.Linear(1024, 256),
			nn.LeakyReLU(),
			nn.Linear(256, 10)  # 10 classes in dataset
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		fc_out = self.fc_block(conv_out)

		return fc_out


class Student(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv_block = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, padding=1),   # -> (N, 16, 32, 32)
			nn.LeakyReLU(),
			nn.Conv2d(16, 16, kernel_size=3, padding=1),  # -> (N, 16, 32, 32)
			nn.LeakyReLU(),
			nn.MaxPool2d(2),                              # -> (N, 16, 16, 16)
			nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> (N, 32, 16, 16)
			nn.LeakyReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),  # -> (N, 32, 16, 16)
			nn.LeakyReLU(),
			nn.MaxPool2d(2)                               # -> (N, 32, 8, 8)
		)
		self.fc_block = nn.Sequential(
			nn.Flatten(),                                 # -> (N, 2048)
			nn.Dropout(),
			nn.Linear(2048, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 10)
		)

	def forward(self, x):
		conv_out = self.conv_block(x)
		fc_out = self.fc_block(conv_out)

		return fc_out
