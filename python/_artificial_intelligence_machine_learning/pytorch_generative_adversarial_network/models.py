"""
Generator and disriminator models

Author: Sam Barba
Created 01/07/2023
"""

from torch import nn, randn_like


class Generator(nn.Module):
	def __init__(self, *, latent_dim):
		super().__init__()
		self.main_block = nn.Sequential(
			nn.ConvTranspose2d(latent_dim, 512, kernel_size=4),
			nn.LeakyReLU(),
			nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(),
			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(),
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
			nn.Tanh()
		)

	def forward(self, x):
		return self.main_block(x)


class Discriminator(nn.Module):
	def __init__(self, *, noise_strength):
		super().__init__()
		self.noise_strength = noise_strength
		self.main_block = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(),
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(),
			nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(),
			nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(),
			nn.Conv2d(512, 1, kernel_size=4),
			nn.Sigmoid()
		)

	def forward(self, x):
		# Gaussian noise injection for regularisation
		noise = randn_like(x) * self.noise_strength

		return self.main_block(x + noise)
