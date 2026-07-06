"""
Generator and disriminator models

Author: Sam Barba
Created 2023-07-01
"""

from torch import nn, randn_like


class Generator(nn.Module):
	def __init__(self, *, latent_dim):
		super().__init__()
		self.main_block = nn.Sequential(
			nn.ConvTranspose2d(latent_dim, 512, kernel_size=4),                            # -> (N, 512, 4, 4)
			nn.ReLU(),
			nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),  # -> (N, 256, 8, 8)
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # -> (N, 128, 16, 16)
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),   # -> (N, 64, 32, 32)
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),                 # -> (N, 3, 64, 64)
			nn.Tanh()
		)

	def forward(self, z):
		return self.main_block(z)


class Discriminator(nn.Module):
	def __init__(self, *, noise_strength):
		super().__init__()
		self.noise_strength = noise_strength
		self.main_block = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),                 # -> (N, 64, 32, 32)
			nn.LeakyReLU(0.2),
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),   # -> (N, 128, 16, 16)
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),
			nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # -> (N, 256, 8, 8)
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),
			nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),  # -> (N, 512, 4, 4)
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2),
			nn.Conv2d(512, 1, kernel_size=4)                                      # -> (N, 1, 1, 1)
		)

	def forward(self, x):
		# Gaussian noise injection for regularisation
		x = x + randn_like(x) * self.noise_strength

		return self.main_block(x)
