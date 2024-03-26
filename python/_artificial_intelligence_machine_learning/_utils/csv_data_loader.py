"""
CSV data loader functionality

Author: Sam Barba
Created 26/03/2024
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch


def load_csv_classification_data(path, train_size=0, val_size=0, test_size=0, x_transform=None, one_hot_y=False, to_tensors=False):
	if train_size == val_size == test_size == 0:
		train_size = 1
	assert np.isclose(train_size + val_size + test_size, 1) and train_size > 0

	df = pd.read_csv(path)
	print(f'\nRaw data:\n\n{df}')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]
	x_to_encode = x.select_dtypes(exclude=np.number).columns
	labels = sorted(y.unique())

	for col in x_to_encode:
		n_unique = x[col].nunique()
		if n_unique == 1:
			# No information from this feature
			x = x.drop(col, axis=1)
		elif n_unique == 2:
			# Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True).astype(int)
		else:
			# Multivariate feature
			one_hot = pd.get_dummies(x[col], prefix=col).astype(int)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)
	features = x.columns

	if one_hot_y:
		# Drop first if binary, keep if multiclass
		y = pd.get_dummies(y, prefix='class', drop_first=len(labels) == 2).astype(int)
	else:
		label_encoder = LabelEncoder()
		y = pd.DataFrame(label_encoder.fit_transform(y), columns=['classification'])

	print(f'\nPreprocessed data:\n\n{pd.concat([x, y], axis=1)}\n')

	x, y = x.to_numpy(), y.to_numpy().squeeze()
	if x_transform:
		x = x_transform.fit_transform(x)
	if to_tensors:
		x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()

	if val_size == 0 and test_size == 0:
		return x, y, labels, features
	else:
		x_train, x_remaining, y_train, y_remaining = train_test_split(x, y, train_size=train_size, stratify=y, random_state=1)
		if val_size == 0 or test_size == 0:
			return x_train, y_train, x_remaining, y_remaining, labels, features
		else:
			rel_train_size = train_size / (train_size + val_size)
			x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=rel_train_size, stratify=y_train, random_state=1)
			return x_train, y_train, x_val, y_val, x_remaining, y_remaining, labels, features


def load_csv_regression_data(path, train_size=0, val_size=0, test_size=0, x_transform=None, to_tensors=False):
	if train_size == val_size == test_size == 0:
		train_size = 1
	assert np.isclose(train_size + val_size + test_size, 1) and train_size > 0

	df = pd.read_csv(path)
	print(f'\nRaw data:\n\n{df}')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]
	x_to_encode = x.select_dtypes(exclude=np.number).columns

	for col in x_to_encode:
		n_unique = x[col].nunique()
		if n_unique == 1:
			# No information from this feature
			x = x.drop(col, axis=1)
		elif n_unique == 2:
			# Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True).astype(int)
		else:
			# Multivariate feature
			one_hot = pd.get_dummies(x[col], prefix=col).astype(int)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)
	features = x.columns

	print(f'\nPreprocessed data:\n\n{pd.concat([x, y], axis=1)}\n')

	x, y = x.to_numpy(), y.to_numpy()
	if x_transform:
		x = x_transform.fit_transform(x)
	if to_tensors:
		x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()

	if val_size == 0 and test_size == 0:
		return x, y, features
	else:
		x_train, x_remaining, y_train, y_remaining = train_test_split(x, y, train_size=train_size, random_state=1)
		if val_size == 0 or test_size == 0:
			return x_train, y_train, x_remaining, y_remaining, features
		else:
			rel_train_size = train_size / (train_size + val_size)
			x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=rel_train_size, random_state=1)
			return x_train, y_train, x_val, y_val, x_remaining, y_remaining, features
