"""
Variational autoencoder class

Author: Sam Barba
Created 08/09/2024
"""

import torch
from torch import nn


class VariationalAutoencoder(nn.Module):
	def __init__(self):
		super().__init__()
		# Input shape (N, 3, 128, 128) (batch size, no. colour channels, height, width)
		self.encoder_block = nn.Sequential(
			nn.Conv2d(3, 16, 3, 2, 1),     # -> (N, 16, 64, 64)
			nn.ReLU(),
			nn.Conv2d(16, 32, 3, 2, 1),    # -> (N, 32, 32, 32)
			nn.ReLU(),
			nn.Conv2d(32, 64, 3, 2, 1),    # -> (N, 64, 16, 16)
			nn.ReLU(),
			nn.Conv2d(64, 128, 3, 2, 1),   # -> (N, 128, 8, 8)
			nn.ReLU(),
			nn.Conv2d(128, 256, 3, 2, 1),  # -> (N, 256, 4, 4)
			nn.ReLU(),
			nn.Conv2d(256, 512, 3, 2, 1),  # -> (N, 512, 2, 2)
			nn.ReLU(),
			nn.Conv2d(512, 1024, 2),       # -> (N, 1024, 1, 1)
			nn.ReLU(),
			nn.Flatten()                   # -> (N, 1024)
		)

		# Instead of 1 latent space, like a regular autoencoder, we have 2:
		# mu (mean) and log_var (log of variance)
		self.fc_mu = nn.Linear(1024, 512)
		self.fc_log_var = nn.Linear(1024, 512)

		self.decoder_block = nn.Sequential(
			nn.Linear(512, 1024),
			nn.ReLU(),
			nn.Unflatten(1, (1024, 1, 1)),             # -> (N, 1024, 1, 1)
			nn.ConvTranspose2d(1024, 512, 2),          # -> (N, 512, 2, 2)
			nn.ReLU(),
			nn.ConvTranspose2d(512, 256, 3),           # -> (N, 256, 4, 4)
			nn.ReLU(),
			nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),  # -> (N, 128, 8, 8)
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),   # -> (N, 64, 16, 16)
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),    # -> (N, 32, 32, 32)
			nn.ReLU(),
			nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),    # -> (N, 16, 64, 64)
			nn.ReLU(),
			nn.ConvTranspose2d(16, 3, 3, 2, 1, 1),     # -> (N, 3, 128, 128)
			nn.Sigmoid()
		)

	def encode(self, x):
		encoded = self.encoder_block(x)
		mu = self.fc_mu(encoded)
		log_var = self.fc_log_var(encoded)

		return mu, log_var

	def reparameterise(self, mu, log_var):
		"""
		Reparameterisation trick (section 2.4 in Auto-Encoding Variational Bayes):
		samples from a Gaussian distribution N(μ, σ²), where N(0, I) is sampled and
		transformed to match the desired mean and variance (mu and log_var)
		"""

		std = torch.exp(0.5 * log_var)  # Standard deviation = root(variance)
		eps = torch.randn_like(std)

		return mu + eps * std

	def forward(self, x):
		mu, log_var = self.encode(x)
		z = self.reparameterise(mu, log_var)
		reconstructed = self.decoder_block(z)

		return reconstructed, mu, log_var

	def loss(self, x_reconstructed, x, mu, log_var, kl_weight):
		"""Appendix B in Auto-Encoding Variational Bayes"""

		# With reduction='sum', the reconstruction loss is roughly the same scale as the KL loss
		reconstruction_loss = nn.functional.mse_loss(x_reconstructed, x, reduction='sum')

		# KL divergence loss
		kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
		kl_loss *= kl_weight  # Weight factor increases from 0 to 1 during training

		vae_loss = (reconstruction_loss + kl_loss) / x.shape[0]  # Normalise across the batch

		return vae_loss
