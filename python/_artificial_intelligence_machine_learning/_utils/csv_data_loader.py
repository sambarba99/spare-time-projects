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


def load_csv_classification_data(
		path, train_size=1, val_size=0, test_size=0, x_transform=None, one_hot_y=False, tensor_device=None
	):
	assert np.isclose(train_size + val_size + test_size, 1) and train_size > 0

	df = pd.read_csv(path)
	print(f'\nRaw data:\n\n{df}\n')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]

	for col in x.columns:
		if x[col].nunique() == 1:
			# No information from this feature
			print(f"Dropping column '{col}' (1 unique value)\n")
			x = x.drop(col, axis=1)

	x_to_encode = x.select_dtypes(exclude=np.number).columns

	for col in x_to_encode:
		num_unique = x[col].nunique()
		if num_unique == 2:
			# Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True, dtype=int)
		else:
			# Multivariate feature
			one_hot = pd.get_dummies(x[col], prefix=col, dtype=int)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)

	features = x.columns
	labels = sorted(y.unique())

	if one_hot_y:
		# Drop first if binary, keep if multiclass
		y = pd.get_dummies(y, prefix='class', drop_first=len(labels) == 2, dtype=int)
	else:
		label_encoder = LabelEncoder()
		y = pd.DataFrame(label_encoder.fit_transform(y), columns=['classification'])

	print(f'Preprocessed data:\n\n{pd.concat([x, y], axis=1)}\n')

	x, y = x.to_numpy(), y.to_numpy().squeeze()
	if x_transform:
		x = x_transform.fit_transform(x)
	if tensor_device:
		x = torch.tensor(x, device=tensor_device).float()
		y = torch.tensor(y, device=tensor_device)
		y = y.float() if len(labels) == 2 else y.long()

	if val_size == test_size == 0:
		return x, y, labels, features

	x_train, x_remaining, y_train, y_remaining = train_test_split(
		x, y, train_size=train_size, stratify=y, random_state=1
	)
	if val_size == 0 or test_size == 0:
		return x_train, y_train, x_remaining, y_remaining, labels, features

	rel_train_size = train_size / (train_size + val_size)
	x_train, x_val, y_train, y_val = train_test_split(
		x_train, y_train, train_size=rel_train_size, stratify=y_train, random_state=1
	)
	return x_train, y_train, x_val, y_val, x_remaining, y_remaining, labels, features


def load_csv_regression_data(path, train_size=1, val_size=0, test_size=0, x_transform=None, tensor_device=None):
	assert np.isclose(train_size + val_size + test_size, 1) and train_size > 0

	df = pd.read_csv(path)
	print(f'\nRaw data:\n\n{df}\n')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]

	for col in x.columns:
		if x[col].nunique() == 1:
			# No information from this feature
			print(f"Dropping column '{col}' (1 unique value)")
			x = x.drop(col, axis=1)

	x_to_encode = x.select_dtypes(exclude=np.number).columns

	for col in x_to_encode:
		num_unique = x[col].nunique()
		if num_unique == 2:
			# Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True, dtype=int)
		else:
			# Multivariate feature
			one_hot = pd.get_dummies(x[col], prefix=col, dtype=int)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)

	features = x.columns

	print(f'Preprocessed data:\n\n{pd.concat([x, y], axis=1)}\n')

	x, y = x.to_numpy(), y.to_numpy().squeeze()
	if x_transform:
		x = x_transform.fit_transform(x)
	if tensor_device:
		x, y = torch.tensor(x, device=tensor_device).float(), torch.tensor(y, device=tensor_device).float()

	if val_size == test_size == 0:
		return x, y, features

	x_train, x_remaining, y_train, y_remaining = train_test_split(x, y, train_size=train_size, random_state=1)
	if val_size == 0 or test_size == 0:
		return x_train, y_train, x_remaining, y_remaining, features

	rel_train_size = train_size / (train_size + val_size)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=rel_train_size, random_state=1)
	return x_train, y_train, x_val, y_val, x_remaining, y_remaining, features
