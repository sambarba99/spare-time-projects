"""
Autoencoder for tabular data

Author: Sam Barba
Created 05/07/2023
"""

from torch import nn


class TabularAutoencoder(nn.Module):
	def __init__(self, num_features_in, num_features_out):
		super().__init__()
		self.encoder_block = nn.Sequential(
			nn.Linear(num_features_in, 32),
			nn.Tanh(),
			nn.Linear(32, 16),
			nn.Tanh(),
			nn.Linear(16, 8),
			nn.Tanh(),
			nn.Linear(8, 4),
			nn.Tanh(),
			nn.Linear(4, num_features_out)
		)
		self.decoder_block = nn.Sequential(
			nn.Linear(num_features_out, 4),
			nn.Tanh(),
			nn.Linear(4, 8),
			nn.Tanh(),
			nn.Linear(8, 16),
			nn.Tanh(),
			nn.Linear(16, 32),
			nn.Tanh(),
			nn.Linear(32, num_features_in)
		)

	def forward(self, x):
		encoded = self.encoder_block(x)
		decoded = self.decoder_block(encoded)
		return decoded
