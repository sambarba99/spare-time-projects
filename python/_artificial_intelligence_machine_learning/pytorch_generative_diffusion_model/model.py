"""
UNet-based diffusion model class

Author: Sam Barba
Created 28/05/2024
"""

import torch
from torch import nn


class DownBlock(nn.Module):
	"""Downsamples the input via convolution and max-pooling"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv_block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.LeakyReLU()
		)
		self.pool = nn.MaxPool2d(2)

	def forward(self, x):
		x_conv = self.conv_block(x)
		pooled = self.pool(x_conv)

		return x_conv, pooled


class UpBlock(nn.Module):
	"""
	Upsamples the input via a transposed convolution, concatenation
	with the corresponding skip connection, and a convolution
	"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv_transpose_block = nn.Sequential(
			nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
			nn.LeakyReLU()
		)
		self.conv_block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.LeakyReLU()
		)

	def forward(self, x, skip_connection):
		upsampled = self.conv_transpose_block(x)
		x_concat = torch.cat([upsampled, skip_connection], dim=1)
		x_conv = self.conv_block(x_concat)

		return x_conv


class SelfAttention(nn.Module):
	"""
	Self attention class (multi-headed if num_heads > 1)
	Source: https://arxiv.org/pdf/1706.03762
	"""

	def __init__(self, *, num_heads, num_channels):
		super().__init__()
		assert num_channels % num_heads == 0

		self.query_conv = nn.Conv2d(num_channels, num_channels, kernel_size=1)
		self.key_conv = nn.Conv2d(num_channels, num_channels, kernel_size=1)
		self.value_conv = nn.Conv2d(num_channels, num_channels, kernel_size=1)

		self.num_heads = num_heads
		self.head_dim = num_channels // self.num_heads
		self.scale = 1 / (self.head_dim ** 0.5)
		self.softmax = nn.Softmax(dim=-1)
		self.gamma = nn.Parameter(torch.zeros(1))

	def forward(self, x):
		b, c, h, w = x.shape

		if self.num_heads == 1:
			# Generate query, key, value matrices
			proj_query = self.query_conv(x).view(b, -1, h * w)
			proj_key = self.key_conv(x).view(b, -1, h * w)
			proj_value = self.value_conv(x).view(b, -1, h * w)

			# Compute attention
			energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key) * self.scale
			attention = self.softmax(energy)

			# Apply attention to value matrix
			out = torch.bmm(proj_value, attention.permute(0, 2, 1))
			out = out.view(b, c, h, w)
		else:
			# Generate query, key, value matrices
			proj_query = self.query_conv(x).view(b, self.num_heads, self.head_dim, h * w)
			proj_key = self.key_conv(x).view(b, self.num_heads, self.head_dim, h * w)
			proj_value = self.value_conv(x).view(b, self.num_heads, self.head_dim, h * w)

			# Compute attention
			energy = torch.einsum('bnhd,bmhd->bhnm', proj_query, proj_key) * self.scale
			attention = self.softmax(energy)

			# Apply attention to value matrix
			out = torch.einsum('bhnm,bmhd->bnhd', attention, proj_value)
			out = out.contiguous().view(b, c, h, w)

		out = self.gamma * out + x

		return out


class DDPM(nn.Module):
	"""
	The main UNet-based architecture that stacks DownBlocks and UpBlocks with concatenation-based
	skip connections between the corresponding layers. The timestep positional encodings are
	inspired by those used in transformers (https://arxiv.org/pdf/1706.03762), and are useful for
	representing time information in the diffusion process.
	"""

	def __init__(self, *, num_timesteps, encoding_dim, device):
		super().__init__()

		# Timestep positional encoding
		assert encoding_dim % 2 == 0
		even_indices = torch.arange(0, encoding_dim, 2)
		log_term = torch.log(torch.tensor(10000)) / encoding_dim
		div_term = torch.exp(even_indices * -log_term)
		timesteps = torch.arange(num_timesteps).unsqueeze(dim=1)
		self.pe_matrix = torch.zeros(num_timesteps, encoding_dim, device=device)
		self.pe_matrix[:, 0::2] = torch.sin(timesteps * div_term)
		self.pe_matrix[:, 1::2] = torch.cos(timesteps * div_term)

		self.time_positional_encoding = nn.Sequential(
			nn.Linear(encoding_dim, encoding_dim),
			nn.GELU(),
			nn.Linear(encoding_dim, encoding_dim),
			nn.GELU()
		)

		# Encoding/downsampling path
		self.encoder1 = DownBlock(3 + encoding_dim, 64)  # 3 input colour channels + timestep encodings
		self.encoder2 = DownBlock(64, 128)
		self.encoder3 = DownBlock(128, 256)
		self.encoder4 = DownBlock(256, 512)

		# Bottleneck
		self.middle = nn.Sequential(
			nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
			nn.LeakyReLU(),
			nn.GroupNorm(num_groups=8, num_channels=1024),
			SelfAttention(num_heads=8, num_channels=1024)
		)

		# Decoding/upsampling path
		self.decoder1 = UpBlock(1024, 512)
		self.decoder2 = UpBlock(512, 256)
		self.decoder3 = UpBlock(256, 128)
		self.decoder4 = UpBlock(128, 64)

		self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

		self.to(device)

	def forward(self, x, t):
		t_enc = self.time_positional_encoding(self.pe_matrix[t])  # -> (N, encoding_dim)

		# Match spatial dimensions of the input, and concatenate to x along the feature dimension
		_, _, h, w = x.shape
		t_enc = t_enc[..., None, None]      # -> (N, encoding_dim, 1, 1)
		t_enc = t_enc.expand(-1, -1, h, w)  # -> (N, encoding_dim, 64, 64)
		xt = torch.cat([x, t_enc], dim=1)   # -> (N, 3 + encoding_dim, 64, 64)

		skip1, enc1 = self.encoder1(xt)    # -> (N, 64, 64, 64) (N, 64, 32, 32)
		skip2, enc2 = self.encoder2(enc1)  # -> (N, 128, 32, 32) (N, 128, 16, 16)
		skip3, enc3 = self.encoder3(enc2)  # -> (N, 256, 16, 16) (N, 256, 8, 8)
		skip4, enc4 = self.encoder4(enc3)  # -> (N, 512, 8, 8) (N, 512, 4, 4)

		mid = self.middle(enc4)            # -> (N, 1024, 4, 4)

		dec1 = self.decoder1(mid, skip4)   # -> (N, 512, 8, 8)
		dec2 = self.decoder2(dec1, skip3)  # -> (N, 256, 16, 16)
		dec3 = self.decoder3(dec2, skip2)  # -> (N, 128, 32, 32)
		dec4 = self.decoder4(dec3, skip1)  # -> (N, 64, 64, 64)

		out = self.final_conv(dec4)        # -> (N, 3, 64, 64)

		return out
