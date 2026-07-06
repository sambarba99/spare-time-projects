"""
Movie review classifier

Author: Sam Barba
Created 2024-03-26
"""

from torch import nn


class MovieReviewClf(nn.Module):
	def __init__(self, *, vocab_size, embedding_dim, hidden_dim):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.fc_block = nn.Sequential(
			nn.Linear(embedding_dim, hidden_dim),
			nn.LeakyReLU(),
			nn.Linear(hidden_dim, hidden_dim // 2),
			nn.LeakyReLU(),
			nn.Linear(hidden_dim // 2, 1)
		)

	def forward(self, x):
		embedded = self.embedding(x)   # -> (N, sequence_len, embedding_dim)
		pooled = embedded.mean(dim=1)  # -> (N, embedding_dim)
		out = self.fc_block(pooled)    # -> (N, 1)

		return out
