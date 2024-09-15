"""
Bayesian hyperparameter optimisation demo (on Boston housing dataset)

Author: Sam Barba
Created 15/09/2024
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from _utils.csv_data_loader import load_csv_regression_data
from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping


plt.rcParams['figure.figsize'] = (6, 4)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', None)

DATASET_PATH = 'C:/Users/Sam/Desktop/projects/datasets/boston_housing.csv'
NUM_FEATURES = 13
NUM_OUTPUTS = 1
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_OPTIMISATION_ITERS = 25


class CustomNN:
	def __init__(self, num_hidden_layers, nodes_per_hidden_layer, activation_type, dropout_rate, learning_rate):
		torch.manual_seed(1)

		layers = [nn.Linear(NUM_FEATURES, nodes_per_hidden_layer)]

		for _ in range(num_hidden_layers - 1):
			layers.append(get_activation_layer(activation_type))
			layers.append(nn.Dropout(dropout_rate))
			layers.append(nn.Linear(nodes_per_hidden_layer, nodes_per_hidden_layer))

		layers.append(get_activation_layer(activation_type))
		layers.append(nn.Dropout(dropout_rate))
		layers.append(nn.Linear(nodes_per_hidden_layer, NUM_OUTPUTS))

		self.model = nn.Sequential(*layers)
		self.optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
		self.loss_func = nn.MSELoss()
		self.early_stopping = EarlyStopping(patience=10, min_delta=0, mode='min')

	def fit(self, train_loader, val_loader):
		for _ in range(NUM_EPOCHS):
			self.model.train()
			for x_train, y_train in train_loader:
				y_pred = self.model(x_train).squeeze()
				loss = self.loss_func(y_pred, y_train)

				self.optimiser.zero_grad()
				loss.backward()
				self.optimiser.step()

			self.model.eval()
			x_val, y_val = next(iter(val_loader))
			with torch.inference_mode():
				y_val_pred = self.model(x_val).squeeze()
			val_loss = self.loss_func(y_val_pred, y_val).item()

			if self.early_stopping(val_loss, self.model.state_dict()):
				break

		self.model.load_state_dict(self.early_stopping.best_weights)  # Restore best weights

	def test(self, test_loader):
		self.model.eval()
		x_test, y_test = next(iter(test_loader))
		with torch.inference_mode():
			y_test_pred = self.model(x_test).squeeze()
		test_loss = self.loss_func(y_test_pred, y_test).item()

		return test_loss


def get_activation_layer(activation_type):
	match activation_type:
		case 'relu':
			return nn.ReLU()
		case 'leakyrelu':
			return nn.LeakyReLU()
		case 'elu':
			return nn.ELU()
		case 'tanh':
			return nn.Tanh()
		case _:
			raise ValueError


def objective(params):
	model = CustomNN(*params)
	model.fit(train_loader, val_loader)
	test_loss = model.test(test_loader)

	return test_loss


def print_progress(res):
	params = res.x_iters[-1]   # Parameters tested in the latest iteration
	score = res.func_vals[-1]  # Corresponding objective function value
	objective_scores.append(score)
	print(f'Iteration {len(res.x_iters)}/{NUM_OPTIMISATION_ITERS} | params: {params} | score: {score}')


if __name__ == '__main__':
	# 1. Load data

	x_train, y_train, x_val, y_val, x_test, y_test, _ = load_csv_regression_data(
		DATASET_PATH,
		train_size=0.7,
		val_size=0.2,
		test_size=0.1,
		x_transform=StandardScaler(),
		tensor_device='cpu'
	)

	train_dataset = CustomDataset(x_train, y_train)
	val_dataset = CustomDataset(x_val, y_val)
	test_dataset = CustomDataset(x_test, y_test)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
	val_loader = DataLoader(val_dataset, batch_size=len(x_val))
	test_loader = DataLoader(test_dataset, batch_size=len(x_test))

	# 2. Define the hyperparameter search space

	search_space = [
		Integer(1, 4),                                      # Num hidden layers: int from 1-4
		Categorical([8, 16, 32, 64]),                       # Nodes per hidden layer
		Categorical(['relu', 'leakyrelu', 'elu', 'tanh']),  # Activation type
		Real(0, 0.5),                                       # Dropout rate: real from 0-0.5
		Real(1e-4, 0.01, prior='log-uniform')               # Learning rate: real from 0.0001-0.01
	]

	# 3. Run Bayesian optimisation

	objective_scores = []
	print('Running Bayesian optimisation...\n')

	res = gp_minimize(
		func=objective,
		dimensions=search_space,
		n_calls=NUM_OPTIMISATION_ITERS,
		callback=[print_progress],
		random_state=1
	)

	print('\nBest hyperparameters:')
	print(f'\tNum hidden layers: {res.x[0]}')
	print(f'\tNodes per hidden layer: {res.x[1]}')
	print(f'\tActivation type: {res.x[2]}')
	print(f'\tDropout rate: {res.x[3]:.2f}')
	print(f'\tLearning rate: {res.x[4]:.2e}')
	print(f'\tBest test loss (MSE): {res.fun:.2f}')

	custom_nn = CustomNN(*res.x)
	print('\nModel with best params:\n')
	print(custom_nn.model)

	running_minimum_objective = np.minimum.accumulate(objective_scores)
	plt.plot(range(1, NUM_OPTIMISATION_ITERS + 1), running_minimum_objective)
	plt.scatter(range(1, NUM_OPTIMISATION_ITERS + 1), running_minimum_objective)
	plt.xlabel('Num calls $n$')
	plt.ylabel('Min $f(x)$ after $n$ calls')
	plt.title('Convergence plot')
	plt.show()
