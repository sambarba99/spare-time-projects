"""
Movie review classifier

Author: Sam Barba
Created 26/03/2024
"""

from torch import mean, nn

from _utils.model_architecture_plots import plot_model_manual


def plot_review_classifier(sequence_len, embedding_len, hidden_len):
	nodes = [
		{'layer_name': 'sequential', 'layer_type': 'Embedding', 'input_shape': (sequence_len,), 'output_shape': (sequence_len, embedding_len)},
		{'layer_name': 'sequential', 'layer_type': 'torch.mean', 'output_shape': (embedding_len,)},
		{'layer_name': 'sequential', 'layer_type': 'Linear', 'output_shape': (hidden_len,), 'activation': 'LeakyReLU'},
		{'layer_name': 'sequential', 'layer_type': 'Linear', 'output_shape': (hidden_len // 2,), 'activation': 'LeakyReLU'},
		{'layer_name': 'sequential', 'layer_type': 'Linear', 'output_shape': (1,)}
	]
	plot_model_manual(nodes=nodes)


class MovieReviewClf(nn.Module):
	def __init__(self, *, vocab_size, embedding_len, hidden_len):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_len)
		self.fc1 = nn.Linear(embedding_len, hidden_len)
		self.fc2 = nn.Linear(hidden_len, hidden_len // 2)
		self.fc3 = nn.Linear(hidden_len // 2, 1)
		self.leaky_relu = nn.LeakyReLU()

	def forward(self, x):
		embedded = self.embedding(x)                  # -> (N, sequence_len, embedding_len)
		pooled = mean(embedded, dim=1)                # -> (N, embedding_len)
		fc1_out = self.leaky_relu(self.fc1(pooled))   # -> (N, hidden_len)
		fc2_out = self.leaky_relu(self.fc2(fc1_out))  # -> (N, hidden_len // 2)
		fc3_out = self.fc3(fc2_out)                   # -> (N, 1)

		return fc3_out.squeeze()
