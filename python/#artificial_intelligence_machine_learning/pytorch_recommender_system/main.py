"""
Recommender system demo using matrix factorisation via PyTorch embeddings

Author: Sam Barba
Created 10/11/2023
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from early_stopping import EarlyStopping


torch.manual_seed(1)

K = 8                # No. latent features (embedding size)
N_EPOCHS = 1000      # No. training epochs
BATCH_SIZE = 50_000  # Training batch size


class MatrixFactorisationModel(nn.Module):
	def __init__(self, n_users, n_shows, embedding_dim):
		super().__init__()
		# Each user is represented as a vector of fixed size (embedding_dim).
		# This vector is the "embedding" for that user. The elements of this
		# vector capture latent features about the user's preferences.
		self.user_embeddings = nn.Embedding(n_users, embedding_dim)
		# Similarly, each show has a vector which is the embedding
		# that captures its latent features.
		self.show_embeddings = nn.Embedding(n_shows, embedding_dim)
		self.user_bias = nn.Embedding(n_users, 1)
		self.show_bias = nn.Embedding(n_shows, 1)

	def forward(self, user_ids, show_ids):
		user_embeds = self.user_embeddings(user_ids)
		show_embeds = self.show_embeddings(show_ids)
		user_bias = self.user_bias(user_ids)
		show_bias = self.show_bias(show_ids)

		prediction = torch.sum(user_embeds * show_embeds, dim=1, keepdim=True)
		prediction += user_bias + show_bias

		return prediction.squeeze()


if __name__ == '__main__':
	# 1. Get DataFrame and print some statistics

	df = pd.read_csv('C:/Users/Sam/Desktop/Projects/datasets/show_ratings.csv')

	print(f'\nRaw data:\n\n{df}\n')
	print(df.describe())
	print(f'\nNo. unique values:\n\n{df.nunique()}')
	print('\nUnique ratings:', dict(df['rating'].value_counts()))

	mean_num_ratings_per_user = np.mean(df.groupby(['user_id']).count()['show_id'])
	print(f'\nMean no. ratings per user: {mean_num_ratings_per_user:.2f}')

	# 2. Create train, validation, and test sets (0.7:0.2:0.1)

	train_df, test_df = train_test_split(df, train_size=0.9, random_state=1)
	train_df, val_df = train_test_split(train_df, train_size=0.78, random_state=1)

	# Encode training DF with continuous user and show IDs, as we need continuous
	# IDs to index into the embedding matrix and access each user/item embedding

	for col in ['user_id', 'show_id']:
		keys = train_df[col].unique()
		key_to_id = {key: idx for idx, key in enumerate(keys)}
		train_df[col] = np.array([key_to_id[i] for i in train_df[col]])

	n_users = train_df['user_id'].nunique()
	n_shows = train_df['show_id'].nunique()

	# Encode val_df with the same encoding as train_df (Cold Start problem: cannot make predictions
	# for users and items not in the training data, as there are no embeddings for them)

	train_user_ids = train_df['user_id'].to_numpy()
	train_show_ids = train_df['show_id'].to_numpy()
	val_df_idx = val_df['user_id'].isin(train_user_ids) \
		& val_df['show_id'].isin(train_show_ids)
	val_df = val_df[val_df_idx]
	val_df['user_id'] = np.array([train_user_ids[i] for i in val_df['user_id']])
	val_df['show_id'] = np.array([train_show_ids[i] for i in val_df['show_id']])

	x_train = torch.from_numpy(train_df[['user_id', 'show_id']].to_numpy())
	y_train = torch.from_numpy(train_df['rating'].to_numpy()).float()
	x_val = torch.from_numpy(val_df[['user_id', 'show_id']].to_numpy())
	y_val = torch.from_numpy(val_df['rating'].to_numpy()).float()

	# 3. Train embeddings

	model = MatrixFactorisationModel(n_users, n_shows, K)
	mae_loss = nn.L1Loss()

	if os.path.exists('./model.pth'):
		model.load_state_dict(torch.load('./model.pth'))
	else:
		loss_func = nn.MSELoss()
		optimiser = torch.optim.Adam(model.parameters())  # LR = 1e-3
		early_stopping = EarlyStopping(patience=10, min_delta=0, mode='min')

		for epoch in range(1, N_EPOCHS + 1):
			for i in range(0, len(x_train), BATCH_SIZE):
				x_batch = x_train[i:i + BATCH_SIZE]
				y_batch = y_train[i:i + BATCH_SIZE]

				y_pred = model(x_batch[:, 0], x_batch[:, 1])
				loss = loss_func(y_pred, y_batch)

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

			with torch.inference_mode():
				val_pred = model(x_val[:, 0], x_val[:, 1])
				val_loss = mae_loss(val_pred, y_val).item()

			if epoch % 10 == 0:
				print(f'Epoch {epoch}/{N_EPOCHS}: val MAE = {val_loss}')

			if early_stopping(val_loss, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './model.pth')

	# 4. Demo model on train and test sets

	with torch.inference_mode():
		train_pred = model(x_train[:, 0], x_train[:, 1])
		train_loss = mae_loss(train_pred, y_train)

	train_df['rating_prediction'] = train_pred
	print(f'\nTrain MAE: {train_loss.item()}')
	print(f'\n{train_df}')

	# Encode test_df with the same encoding as train_df (Cold Start problem: cannot make predictions
	# for users and items not in the training data, as there are no embeddings for them)

	test_df_idx = test_df['user_id'].isin(train_user_ids) \
		& test_df['show_id'].isin(train_show_ids)
	test_df = test_df[test_df_idx]
	test_df['user_id'] = np.array([train_user_ids[i] for i in test_df['user_id']])
	test_df['show_id'] = np.array([train_show_ids[i] for i in test_df['show_id']])

	x_test = torch.from_numpy(test_df[['user_id', 'show_id']].to_numpy())
	y_test = torch.from_numpy(test_df['rating'].to_numpy()).float()

	with torch.inference_mode():
		test_pred = model(x_test[:, 0], x_test[:, 1])
		test_loss = mae_loss(test_pred, y_test)

	test_df['rating_prediction'] = test_pred
	print(f'\nTest MAE: {test_loss.item()}')
	print(f'\n{test_df}')

	user_0_embed = model.user_embeddings(torch.tensor([0])).detach()
	user_0_bias = model.user_bias(torch.tensor([0])).detach()
	show_1_embed = model.show_embeddings(torch.tensor([1])).detach()
	show_1_bias = model.show_bias(torch.tensor([1])).detach()
	result = user_0_embed[0].dot(show_1_embed[0]) + user_0_bias + show_1_bias

	print('\nEmbedding for user 0:', user_0_embed)
	print('Bias for user 0:', user_0_bias)
	print('Embedding for show 1:', show_1_embed)
	print('Bias for show 1:', show_1_bias)

	print('\nUser 0 rating for show 1:')
	print('\tuser_0_embed.dot(show_1_embed) + user_0_bias + show_1_bias')
	print('\t=', result.item())
