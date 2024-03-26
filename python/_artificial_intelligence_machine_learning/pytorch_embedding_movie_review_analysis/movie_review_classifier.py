"""
Movie review classifier

Author: Sam Barba
Created 26/03/2024
"""

from torch import mean, nn

from _utils.model_architecture_plots import plot_model_manual


def plot_review_classifier():
	nodes = [
		{'layer_name': 'sequential', 'layer_type': 'Embedding', 'input_shape': (42231,), 'output_shape': (200, 256), 'activation': 'LeakyReLU'},
		{'layer_name': 'sequential', 'layer_type': 'Avg Pooling', 'input_shape': (200, 256), 'output_shape': (256,)},
		{'layer_name': 'sequential', 'layer_type': 'Linear', 'input_shape': (256,), 'output_shape': (128,), 'activation': 'LeakyReLU'},
		{'layer_name': 'sequential', 'layer_type': 'Linear', 'input_shape': (128,), 'output_shape': (1,), 'activation': 'Sigmoid'}
	]
	edges = [(i, i + 1) for i in range(len(nodes) - 1)]
	plot_model_manual(nodes, edges, (42231,))


class MovieReviewClf(nn.Module):
	def __init__(self, *, vocab_size, embedding_dim, hidden_dim):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.fc1 = nn.Linear(embedding_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, 1)
		self.leaky_relu = nn.LeakyReLU()
		self.sigmoid = nn.Sigmoid()
		plot_review_classifier()

	def forward(self, x):
		embedded = self.leaky_relu(self.embedding(x))
		pooled = mean(embedded, dim=1)
		fc1_out = self.leaky_relu(self.fc1(pooled))
		fc2_out = self.sigmoid(self.fc2(fc1_out))
		return fc2_out.squeeze()
