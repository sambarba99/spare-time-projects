"""
Autoencoder for tabular data

Author: Sam Barba
Created 05/07/2023
"""

from torch import nn


class TabularAutoencoder(nn.Module):
	def __init__(self, n_features_in, n_features_out):
		super(TabularAutoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(n_features_in, 32),
			nn.Tanh(),
			nn.Linear(32, 16),
			nn.Tanh(),
			nn.Linear(16, 8),
			nn.Tanh(),
			nn.Linear(8, 4),
			nn.Tanh(),
			nn.Linear(4, n_features_out)
		)

		self.decoder = nn.Sequential(
			nn.Linear(n_features_out, 4),
			nn.Tanh(),
			nn.Linear(4, 8),
			nn.Tanh(),
			nn.Linear(8, 16),
			nn.Tanh(),
			nn.Linear(16, 32),
			nn.Tanh(),
			nn.Linear(32, n_features_in)
		)


	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded
