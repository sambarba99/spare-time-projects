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
import optuna
import torch
from torch.utils.data import DataLoader

from _utils.csv_data_loader import load_csv_regression_data
from _utils.custom_dataset import CustomDataset
from custom_nn import CustomNN


plt.rcParams['figure.figsize'] = (8, 5)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', None)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATASET_PATH = 'C:/Users/sam/Desktop/projects/datasets/boston_housing.csv'
BATCH_SIZE = 32
NUM_OPTIMISATION_ITERS = 25


def scikit_objective(params):
	torch.manual_seed(1)

	model = CustomNN(*params)
	val_loss = model.fit(train_loader, val_loader)

	return val_loss


def optuna_objective(trial):
	torch.manual_seed(1)

	num_hidden_layers = trial.suggest_int('Num hidden layers', 1, 4)
	nodes_per_hidden_layer = trial.suggest_categorical('Nodes per hidden layer', [8, 16, 32, 64])
	activation_type = trial.suggest_categorical('Activation type', ['relu', 'leakyrelu', 'elu', 'gelu', 'prelu'])
	dropout_rate = trial.suggest_float('Dropout rate', 0, 0.5)
	learning_rate = trial.suggest_float('Learning rate', 1e-5, 1e-2, log=True)
	params = [num_hidden_layers, nodes_per_hidden_layer, activation_type, dropout_rate, learning_rate]

	model = CustomNN(*params)
	val_loss = model.fit(train_loader, val_loader)

	objective_scores.append(val_loss)
	print(f'Iteration {len(objective_scores)}/{NUM_OPTIMISATION_ITERS} | params: {params} | score: {val_loss}')

	return val_loss


def print_progress(res):
	params = res.x_iters[-1]   # Parameters tested in the latest iteration
	score = res.func_vals[-1]  # Corresponding objective function value
	objective_scores.append(score)
	params = [
		int(p) if isinstance(p, np.integer)
		else float(p) if isinstance(p, np.floating)
		else str(p) if isinstance(p, np.str_)
		else p
		for p in params
	]
	print(f'Iteration {len(res.x_iters)}/{NUM_OPTIMISATION_ITERS} | params: {params} | score: {score}')


if __name__ == '__main__':
	# Load data

	x_train, y_train, x_val, y_val, _ = load_csv_regression_data(
		DATASET_PATH,
		train_size=0.8,
		val_size=0.2,
		x_transform=StandardScaler(),
		output_as_tensor=True
	)

	train_dataset = CustomDataset(x_train, y_train)
	val_dataset = CustomDataset(x_val, y_val)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=len(x_val))

	# Define the hyperparameter search space (for scikit-optimize)

	search_space = [
		Integer(1, 4),                                               # Num hidden layers
		Categorical([8, 16, 32, 64]),                                # Nodes per hidden layer
		Categorical(['relu', 'leakyrelu', 'elu', 'gelu', 'prelu']),  # Activation type
		Real(0, 0.5),                                                # Dropout rate
		Real(1e-5, 1e-2, prior='log-uniform')                        # Learning rate (sampled from a logarithmic scale)
	]

	# Run optimisation

	choice = input('Enter 1 for scikit-optimize or 2 for Optuna: ')

	objective_scores = []
	print('\nRunning optimisation...\n')

	if choice == '1':
		res = gp_minimize(
			func=scikit_objective,
			dimensions=search_space,
			n_calls=NUM_OPTIMISATION_ITERS,
			callback=[print_progress],
			random_state=1
		)

		print(f'\nBest val loss (MSE): {res.fun:.2f}')

		print('\nBest hyperparameters:')
		print(f'\tNum hidden layers: {res.x[0]}')
		print(f'\tNodes per hidden layer: {res.x[1]}')
		print(f'\tActivation type: {res.x[2]}')
		print(f'\tDropout rate: {res.x[3]:.2f}')
		print(f'\tLearning rate: {res.x[4]:.2e}')
	else:
		sampler = optuna.samplers.TPESampler(seed=1)
		study = optuna.create_study(direction='minimize', sampler=sampler)
		study.optimize(optuna_objective, n_trials=NUM_OPTIMISATION_ITERS)

		print(f'\nBest val loss (MSE): {study.best_value:.2f}')

		print('\nBest hyperparameters:')
		for k, v in study.best_params.items():
			print(f'\t{k}: {v}')

	running_minimum_objective = np.minimum.accumulate(objective_scores)
	plt.plot(range(1, NUM_OPTIMISATION_ITERS + 1), running_minimum_objective, marker='o')
	plt.xlabel('Num calls $n$')
	plt.ylabel('Min $f(x)$ after $n$ calls')
	plt.title('Convergence plot')
	plt.show()
