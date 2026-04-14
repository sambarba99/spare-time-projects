"""
Visualising the curve fitting of a PyTorch regression model

Author: Sam Barba
Created 14/02/2024
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import torch
from torch import nn


plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
np.random.seed(1)
torch.manual_seed(1)

NUM_DATA_POINTS = 3000
NOISE = 0.25
LEARNING_RATE = 3e-3
NUM_EPOCHS = 200


def plot_val_progress(x_val, y_val_true, y_val_pred, val_rmse, epoch):
	plt.cla()
	plt.gca().set_facecolor('black')
	plt.gcf().set_facecolor('#0d1117')
	plt.scatter(x_val, y_val_true, c='#00ff00', alpha=0.4, label='True val')
	plt.plot(x_val, y_val_pred, c='white', linewidth=2, label='Pred val')
	plt.xlabel('$x$', color='white', fontsize=14)
	plt.ylabel('$y$', color='white', fontsize=14)
	plt.tick_params(colors='white')
	if epoch == 0:
		plt.title(f'Start (random weights)  |  Val RMSE = {val_rmse:.3f}', color='white')
	else:
		plt.title(f'Epoch {epoch}/{NUM_EPOCHS}  |  Val RMSE = {val_rmse:.3f}', color='white')
	legend = plt.legend(facecolor='#808080')
	for handle in legend.legend_handles:
		handle.set_alpha(1)
	# plt.savefig(f'./{epoch:0>3}.png')
	if epoch == NUM_EPOCHS:
		plt.show()
	else:
		plt.draw()
		plt.pause(1e-6)


if __name__ == '__main__':
	# Define an arbitrary function
	x = np.linspace(-1, 1, NUM_DATA_POINTS)
	y = 5 * x ** 3 - x ** 2 - 3 * x + 1  # 5x^3 - x^2 - 3x + 1
	y += np.random.normal(0, NOISE, size=len(x))

	x, y = torch.tensor(x).float(), torch.tensor(y).float()

	x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.9, random_state=1)

	# Sort val data in order of x so it's plottable
	val_idx = np.argsort(x_val)
	x_val = x_val[val_idx]
	y_val = y_val[val_idx]

	# Define model
	model = nn.Sequential(
		nn.Linear(1, 64),
		nn.GELU(),
		nn.Linear(64, 64),
		nn.GELU(),
		nn.Linear(64, 1)
	).cpu()
	loss_func = nn.MSELoss()
	optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	print(f'\nModel:\n{model}\n')

	# Plot start (random weights)
	model.eval()
	with torch.inference_mode():
		y_val_pred = model(x_val.unsqueeze(dim=1)).squeeze()

	val_rmse = root_mean_squared_error(y_val, y_val_pred)
	plot_val_progress(x_val, y_val, y_val_pred, val_rmse, 0)

	# Training loop
	for epoch in range(1, NUM_EPOCHS + 1):
		model.train()
		y_train_pred = model(x_train.unsqueeze(dim=1)).squeeze()
		loss = loss_func(y_train_pred, y_train)

		optimiser.zero_grad()
		loss.backward()
		optimiser.step()

		model.eval()
		with torch.inference_mode():
			y_val_pred = model(x_val.unsqueeze(dim=1)).squeeze()

		val_rmse = root_mean_squared_error(y_val, y_val_pred)
		plot_val_progress(x_val, y_val, y_val_pred, val_rmse, epoch)
