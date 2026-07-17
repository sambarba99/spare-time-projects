"""
Lottery ticket hypothesis demo (unstructured pruning)

Author: Sam Barba
Created 2026-07-17
"""

from copy import deepcopy

from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn


plt.rcParams['figure.figsize'] = (7, 5)
torch.manual_seed(1)


NUM_DATA_POINTS = 1000
NUM_EPOCHS = 500
NUM_PRUNE_ROUNDS = 20
PRUNE_FRACTION = 0.139  # Pruning this amount for 20 rounds yields ~95% final model sparsity


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


def train_model(model, mask):
	model.train()
	optimiser = torch.optim.Adam(model.parameters())  # LR = 1e-3

	for _ in range(NUM_EPOCHS):
		logits = model(x_train)
		loss = loss_func(logits, y_train)

		optimiser.zero_grad()
		loss.backward()
		optimiser.step()

		# Re-apply mask after every optimiser step (pruned weights may have come back from 0)
		if mask is not None:
			apply_mask(model, mask)

	model.eval()
	with torch.inference_mode():
		logits = model(x_test)
	test_loss = loss_func(logits, y_test)

	return test_loss.item()


def create_mask(model, current_mask):
	# Gather all live weights
	live_weights = []

	for name, param in model.named_parameters():
		if 'weight' in name:
			w = param.data.abs()
			if current_mask is None:
				live_weights.append(w.flatten())
			else:
				live_weights.append(w[current_mask[name].bool()])

	live_weights = torch.cat(live_weights)

	# Threshold for pruning
	threshold = torch.quantile(live_weights, PRUNE_FRACTION)

	mask = dict()

	for name, param in model.named_parameters():
		if 'weight' in name:
			if current_mask is None:
				mask[name] = (param.data.abs() > threshold).float()
			else:
				mask[name] = current_mask[name] * (param.data.abs() > threshold).float()

	return mask


def apply_mask(model, mask):
	for name, param in model.named_parameters():
		if name in mask:
			param.data *= mask[name]


def sparsity_stats(mask):
	total = remaining = 0

	for m in mask.values():
		total += m.numel()
		remaining += m.sum().item()

	sparsity = 1 - remaining / total

	return int(remaining), total, sparsity


def plot_weight_matrix(w, round_num):
	fig, ax = plt.subplots()
	mat = ax.matshow(w, cmap='bwr', norm=norm)
	plt.axis('off')
	plt.colorbar(mat, ax=ax)
	plt.title(f'Round {round_num}' if round_num > 0 else 'Start dense model', y=1.02)
	plt.savefig(f'./images/{round_num:0>3}.png')
	plt.close(fig)


if __name__ == '__main__':
	# Define data

	x = torch.linspace(-3, 3, NUM_DATA_POINTS).unsqueeze(dim=1)
	y = torch.sin(2 * x) + 0.3 * x ** 2 + 0.2 * torch.randn_like(x)

	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)

	# Train dense network

	print('\n----- TRAINING DENSE NET -----\n')

	dense_net = make_model()
	initial_state = deepcopy(dense_net.state_dict())
	loss_func = nn.MSELoss()
	test_loss = train_model(dense_net, mask=None)

	norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
	plot_weight_matrix(dense_net.state_dict()['2.weight'], 0)

	print(f'Dense net test loss: {test_loss:.4f}')

	# Iterative pruning

	print('\n----- ITERATIVE MAGNITUDE PRUNING -----\n')

	current_mask = None

	for i in range(NUM_PRUNE_ROUNDS):
		# Update mask
		current_mask = create_mask(dense_net, current_mask)

		# Reset surviving weights to original initialisation
		sparse_net = make_model()
		sparse_net.load_state_dict(initial_state)

		# Apply mask
		apply_mask(sparse_net, current_mask)

		remaining, total, sparsity = sparsity_stats(current_mask)
		test_loss = train_model(sparse_net, mask=current_mask)
		plot_weight_matrix(sparse_net.state_dict()['2.weight'], i + 1)

		# Plot weight matrix of the 2nd linear layer

		print(f'Pruning round {i + 1}/{NUM_PRUNE_ROUNDS}  |  '
			f'Remaining weights: {remaining:,} / {total:,}  |  '
			f'Sparsity: {sparsity:.1%}  |  '
			f'Test loss: {test_loss:.4f}')

		# Continue pruning from this trained sparse model
		dense_net = sparse_net

	# Plot final winning ticket predictions

	sparse_net.eval()
	with torch.inference_mode():
		pred = sparse_net(x_test)

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
	plt.title(f'Lottery Ticket Hypothesis regression demo (unstructured pruning)\nFinal sparsity = {sparsity:.1%}')
	plt.legend()
	plt.show()
