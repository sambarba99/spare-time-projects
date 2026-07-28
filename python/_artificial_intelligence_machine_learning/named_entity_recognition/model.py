"""
NER classifier

Author: Sam Barba
Created 2026-07-28
"""

from torch import nn


class NERClf(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
		self.dropout = nn.Dropout(0.3)
		self.linear = nn.Linear(hidden_dim * 2, num_tags)

	def forward(self, x):
		# x: (N, num_tokens)
		embedded = self.embedding(x)          # -> (N, num_tokens, embedding_dim)
		lstm_out, _ = self.lstm(embedded)     # -> (N, num_tokens, embedding_dim)
		dropout_out = self.dropout(lstm_out)  # -> (N, num_tokens, embedding_dim)
		logits = self.linear(dropout_out)     # -> (N, num_tokens, num_tags)

		return logits
