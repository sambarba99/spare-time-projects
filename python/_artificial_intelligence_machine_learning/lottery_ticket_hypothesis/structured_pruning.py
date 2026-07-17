"""
Lottery ticket hypothesis demo (structured pruning)

Author: Sam Barba
Created 2026-07-17
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from _utils.plotting import plot_torch_model


plt.rcParams['figure.figsize'] = (7, 5)
torch.manual_seed(1)


NUM_DATA_POINTS = 1000
NUM_EPOCHS = 500
NUM_PRUNE_ROUNDS = 10
PRUNE_FRACTION = 0.2  # Removing this amount for 10 rounds yields ~89% final neuron reduction (~99% parameter reduction)


def make_model():
	return nn.Sequential(
		nn.Linear(1, 128),
		nn.LeakyReLU(),
		nn.Linear(128, 128),
		nn.LeakyReLU(),
		nn.Linear(128, 128),
		nn.LeakyReLU(),
		nn.Linear(128, 1)
	)


def train_model(model):
	model.train()
	optimiser = torch.optim.Adam(model.parameters())  # LR = 1e-3

	for _ in range(NUM_EPOCHS):
		logits = model(x_train)
		loss = loss_func(logits, y_train)

		optimiser.zero_grad()
		loss.backward()
		optimiser.step()

	model.eval()
	with torch.inference_mode():
		logits = model(x_test)
	test_loss = loss_func(logits, y_test)

	return test_loss.item()


def prune_layer(layer):
	"""Remove output neurons from a Linear layer"""

	weights = layer.weight.data
	importance = weights.abs().sum(dim=1)  # L1 norm of each neuron (low magnitude outgoing weights = less important)
	keep = int(layer.out_features * (1 - PRUNE_FRACTION))

	_, indices = torch.topk(importance, k=keep)
	indices = indices.sort().values

	new_layer = nn.Linear(layer.in_features, keep)
	new_layer.weight.data = layer.weight.data[indices]
	new_layer.bias.data = layer.bias.data[indices]

	return new_layer, indices


def rebuild_network(model):
	layers = [m for m in model if isinstance(m, nn.Linear)]
	new_layers = []
	previous_indices = None

	for layer in layers[:-1]:
		# Remove neurons
		new_layer, keep_indices = prune_layer(layer)

		# If previous layer was pruned, remove matching input weights
		if previous_indices is not None:
			new_layer.weight.data = new_layer.weight.data[:, previous_indices]
			new_layer.in_features = len(previous_indices)

		new_layers.append(new_layer)
		previous_indices = keep_indices

	# Final output layer

	final_layer = layers[-1]

	if previous_indices is not None:
		final_layer.weight.data = final_layer.weight.data[:, previous_indices]
		final_layer.in_features = len(previous_indices)

	new_layers.append(final_layer)

	rebuilt = []
	for layer in new_layers[:-1]:
		rebuilt.append(layer)
		rebuilt.append(nn.LeakyReLU())
	rebuilt.append(new_layers[-1])

	return nn.Sequential(*rebuilt)


if __name__ == '__main__':
	# Define data

	x = torch.linspace(-3, 3, NUM_DATA_POINTS).unsqueeze(dim=1)
	y = torch.sin(2 * x) + 0.3 * x ** 2 + 0.2 * torch.randn_like(x)

	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)

	# Train dense network

	print('\n----- TRAINING DENSE NET -----\n')

	model = make_model()
	plot_torch_model(model, (1,), out_file='./images/dense_model')

	loss_func = nn.MSELoss()
	dense_params = sum(p.numel() for p in model.parameters())
	test_loss = train_model(model)

	print(f'Dense net params: {dense_params:,}')
	print(f'Dense net test loss: {test_loss:.4f}')

	# Structured pruning

	print('\n----- STRUCTURED PRUNING -----\n')

	for i in range(NUM_PRUNE_ROUNDS):
		model = rebuild_network(model)
		test_loss = train_model(model)
		params = sum(p.numel() for p in model.parameters())

		print(f'Pruning round {i + 1}/{NUM_PRUNE_ROUNDS}  |  '
			f'Remaining params: {params:,} / {dense_params:,}  |  '
			f'Relative size: {(params / dense_params):.1%}  |  '
			f'Test loss: {test_loss:.4f}')

	# Plot final winning ticket predictions

	plot_torch_model(model, (1,), out_file='./images/winning_ticket')

	model.eval()
	with torch.inference_mode():
		pred = model(x_test)

	# Sort test data in order of x so it's plottable
	x_test, y_test = x_test.squeeze(), y_test.squeeze()
	test_idx = torch.argsort(x_test)
	x_test = x_test[test_idx]
	y_test = y_test[test_idx]
	pred = pred[test_idx]

	plt.scatter(x_test, y_test, s=4, color='black', label='Ground truth (test)')
	plt.plot(x_test, pred, color='red', label='Winning ticket prediction')
	plt.axis('scaled')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title(f'Lottery Ticket Hypothesis regression demo (structured pruning)\nFinal relative size = {(params / dense_params):.1%}')
	plt.legend()
	plt.show()
