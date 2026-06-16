"""
Custom neural net class

Author: Sam Barba
Created 2024-09-15
"""

import torch
from torch import nn

from _utils.early_stopping import EarlyStopping


NUM_FEATURES = 13
NUM_OUTPUTS = 1
NUM_EPOCHS = 100


class CustomNN:
	def __init__(self, num_hidden_layers, nodes_per_hidden_layer, activation_type, dropout_rate, learning_rate):
		layers = [nn.Linear(NUM_FEATURES, nodes_per_hidden_layer)]

		for _ in range(num_hidden_layers - 1):
			layers.append(get_activation_layer(activation_type))
			layers.append(nn.Dropout(dropout_rate))
			layers.append(nn.Linear(nodes_per_hidden_layer, nodes_per_hidden_layer))

		layers.append(get_activation_layer(activation_type))
		layers.append(nn.Dropout(dropout_rate))
		layers.append(nn.Linear(nodes_per_hidden_layer, NUM_OUTPUTS))

		self.model = nn.Sequential(*layers)
		self.loss_func = nn.MSELoss()
		self.optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
		self.early_stopping = EarlyStopping(target=self.model, patience=10, mode='min', track_best_weights=False)

	def fit(self, train_loader, val_loader):
		for _ in range(NUM_EPOCHS):
			self.model.train()
			for x_train, y_train in train_loader:
				preds = self.model(x_train).squeeze()
				loss = self.loss_func(preds, y_train)

				self.optimiser.zero_grad()
				loss.backward()
				self.optimiser.step()

			self.model.eval()
			x_val, y_val = next(iter(val_loader))
			with torch.inference_mode():
				preds = self.model(x_val).squeeze()
			val_loss = self.loss_func(preds, y_val).item()

			if self.early_stopping(val_loss):
				break

		return self.early_stopping.best_score


def get_activation_layer(activation_type):
	match activation_type:
		case 'relu':
			return nn.ReLU()
		case 'leakyrelu':
			return nn.LeakyReLU()
		case 'elu':
			return nn.ELU()
		case 'gelu':
			return nn.GELU()
		case 'prelu':
			return nn.PReLU()
		case _:
			raise ValueError
