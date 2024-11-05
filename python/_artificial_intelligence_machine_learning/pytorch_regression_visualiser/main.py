"""
Visualising the curve fitting of a PyTorch regression model

Author: Sam Barba
Created 14/02/2024
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
import torch
from torch import nn


plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
np.random.seed(1)
torch.manual_seed(1)

NUM_HIDDEN_LAYERS = 2
HIDDEN_LAYER_SIZE = 16
LEARNING_RATE = 3e-3
NUM_SPLITS = 5
NUM_EPOCHS = 100


def plot_test_progress(x_test, y_test_true, y_test_pred, epoch_num, split_num):
	mae = mean_absolute_error(y_test_true, y_test_pred)
	plt.cla()
	plt.gca().set_facecolor('black')
	plt.gcf().set_facecolor('#202020')
	plt.scatter(x_test, y_test_true, c='#00ff00', alpha=0.4, label='True test')
	plt.plot(x_test, y_test_pred, c='white', linewidth=2, label='Pred test')
	plt.xlabel('$x$', color='white', fontsize=14)
	plt.ylabel('$y$', color='white', fontsize=14)
	plt.tick_params(colors='white')
	plt.title(f'Epoch {epoch_num}/{NUM_EPOCHS}, split {split_num}/{NUM_SPLITS}\nTest MAE = {mae:.4f}', color='white')
	legend = plt.legend()
	for handle in legend.legend_handles:
		handle.set_alpha(1)
	plt.draw()
	plt.pause(1e-6)


if __name__ == '__main__':
	# 1. Define an arbitrary function

	x = np.linspace(-1, 1, 2000)
	y = 5 * x ** 3 - x ** 2 - 3 * x + 1  # 5x^3 - x^2 - 3x + 1
	noise = 0.5
	y += np.random.uniform(-noise, noise, len(x))

	x, y = torch.tensor(x).float(), torch.tensor(y).float()

	x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, train_size=0.9, random_state=1)

	test_idx = np.argsort(x_test)  # Sort test data in order of x so it's plottable
	x_test = x_test[test_idx]
	y_test = y_test[test_idx]

	# 2. Define model

	layers = []
	for i in range(NUM_HIDDEN_LAYERS):
		layers.append(nn.Linear(1 if i == 0 else HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE))
		layers.append(nn.Tanh())
	layers.append(nn.Linear(HIDDEN_LAYER_SIZE, 1))

	model = nn.Sequential(*layers)
	model.to('cpu')
	loss_func = nn.MSELoss()
	optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	print(f'\nModel:\n{model}\n')

	# 3. Training loop

	kf = KFold(n_splits=NUM_SPLITS)

	for epoch in range(1, NUM_EPOCHS + 1):
		print(f'Epoch {epoch}/{NUM_EPOCHS}')

		for split_num, (train_idx, val_idx) in enumerate(kf.split(x_train_val), start=1):
			x_train = x_train_val[train_idx]
			y_train = y_train_val[train_idx]
			x_val = x_train_val[val_idx]
			y_val = y_train_val[val_idx]

			y_train_pred = model(x_train.unsqueeze(dim=1)).squeeze()
			loss = loss_func(y_train_pred, y_train)

			optimiser.zero_grad()
			loss.backward()
			optimiser.step()

			with torch.inference_mode():
				y_val_pred = model(x_val.unsqueeze(dim=1)).squeeze()
				y_test_pred = model(x_test.unsqueeze(dim=1)).squeeze()

			val_mae = mean_absolute_error(y_val, y_val_pred)
			print(f'\tSplit {split_num}/{NUM_SPLITS}  |  Val MAE = {val_mae:.4f}')
			plot_test_progress(x_test, y_test, y_test_pred, epoch, split_num)

	plt.show()
