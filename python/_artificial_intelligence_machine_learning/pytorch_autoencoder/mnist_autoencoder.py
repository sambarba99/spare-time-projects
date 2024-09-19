"""
Autoencoder for MNIST data

Author: Sam Barba
Created 17/06/2023
"""

from torch import nn


class MNISTAutoencoder(nn.Module):
	def __init__(self):
		super().__init__()
		# Input shape (N, 1, 28, 28) (batch size, no. colour channels, height, width)
		self.encoder_block = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # -> (N, 16, 14, 14)
			nn.Tanh(),
			nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (N, 32, 7, 7)
			nn.Tanh(),
			nn.Conv2d(32, 64, kernel_size=7),                       # -> (N, 64, 1, 1)
			nn.Tanh(),
			nn.Flatten(),                                           # -> (N, 64)
			nn.Linear(64, 16),
			nn.Tanh(),
			nn.Linear(16, 8),
			nn.Tanh(),
			nn.Linear(8, 2)  # Latent space shape = (N, 2) (plottable on xy axes)
		)
		# Input shape (N, 2) (392x compression!)
		self.decoder_block = nn.Sequential(
			nn.Linear(2, 8),
			nn.Tanh(),
			nn.Linear(8, 16),
			nn.Tanh(),
			nn.Linear(16, 64),
			nn.Tanh(),
			nn.Unflatten(1, (64, 1, 1)),                                                       # -> (N, 64, 1, 1)
			nn.ConvTranspose2d(64, 32, kernel_size=7),                                         # -> (N, 32, 7, 7)
			nn.Tanh(),
			nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (N, 16, 14, 14)
			nn.Tanh(),
			nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (N, 1, 28, 28)
			nn.Sigmoid()
		)

	def forward(self, x):
		encoded = self.encoder_block(x)
		reconstructed = self.decoder_block(encoded)

		return reconstructed
