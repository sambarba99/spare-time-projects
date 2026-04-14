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
		path,
		*,
		header='infer',
		train_size=1,
		val_size=0,
		test_size=0,
		drop_useless_features=True,
		x_transform=None,
		y_col_pos=-1,
		one_hot_y=False,
		output_as_tensor=False
	):
	assert header in ('infer', None)
	assert train_size > 0
	assert val_size >= 0 and test_size >= 0
	assert np.isclose(train_size + val_size + test_size, 1)

	df = pd.read_csv(path, header=header)
	print(f'\nRaw data:\n\n{df}\n')

	x = df.drop(df.columns[y_col_pos], axis=1)
	y = df.iloc[:, y_col_pos]

	if drop_useless_features:
		drop_columns = [c for c in x.columns if x[c].nunique() == 1]
		if drop_columns:
			print(f'Dropping these columns (1 unique value): {drop_columns}\n')
			x = x.drop(drop_columns, axis=1)

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

	x, y = x.to_numpy(dtype=np.float32), y.to_numpy().squeeze()

	if val_size == test_size == 0:
		if x_transform:
			if callable(x_transform):
				# x_transform is a custom Python func
				x = x_transform(x)
			else:
				# sklearn transform e.g. MinMaxScaler, StandardScaler
				x = x_transform.fit_transform(x)

		if output_as_tensor:
			x = torch.tensor(x).float()
			y = torch.tensor(y).float() if len(labels) == 2 else torch.tensor(y).long()

		return x, y, labels, features

	x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, train_size=train_size, stratify=y, random_state=1)
	if val_size == 0 or test_size == 0:
		if x_transform:
			if callable(x_transform):
				x_train = x_transform(x_train)
				x_tmp = x_transform(x_tmp)
			else:
				# Fit only on x_train
				x_train = x_transform.fit_transform(x_train)
				x_tmp = x_transform.transform(x_tmp)

		if output_as_tensor:
			x_train = torch.tensor(x_train).float()
			x_tmp = torch.tensor(x_tmp).float()
			y_train = torch.tensor(y_train).float() if len(labels) == 2 else torch.tensor(y_train).long()
			y_tmp = torch.tensor(y_tmp).float() if len(labels) == 2 else torch.tensor(y_tmp).long()

		return x_train, y_train, x_tmp, y_tmp, labels, features

	val_size /= (val_size + test_size)
	x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, train_size=val_size, stratify=y_tmp, random_state=1)
	if x_transform:
		if callable(x_transform):
			x_train = x_transform(x_train)
			x_val = x_transform(x_val)
			x_test = x_transform(x_test)
		else:
			# Fit only on x_train
			x_train = x_transform.fit_transform(x_train)
			x_val = x_transform.transform(x_val)
			x_test = x_transform.transform(x_test)

	if output_as_tensor:
		x_train = torch.tensor(x_train).float()
		x_val = torch.tensor(x_val).float()
		x_test = torch.tensor(x_test).float()
		y_train = torch.tensor(y_train).float() if len(labels) == 2 else torch.tensor(y_train).long()
		y_val = torch.tensor(y_val).float() if len(labels) == 2 else torch.tensor(y_val).long()
		y_test = torch.tensor(y_test).float() if len(labels) == 2 else torch.tensor(y_test).long()

	return x_train, y_train, x_val, y_val, x_test, y_test, labels, features


def load_csv_regression_data(
		path,
		*,
		header='infer',
		train_size=1,
		val_size=0,
		test_size=0,
		drop_useless_features=True,
		x_transform=None,
		y_col_pos=-1,
		output_as_tensor=False
	):
	assert header in ('infer', None)
	assert train_size > 0
	assert val_size >= 0 and test_size >= 0
	assert np.isclose(train_size + val_size + test_size, 1)

	df = pd.read_csv(path, header=header)
	print(f'\nRaw data:\n\n{df}\n')

	x = df.drop(df.columns[y_col_pos], axis=1)
	y = df.iloc[:, y_col_pos]

	if drop_useless_features:
		drop_columns = [c for c in x.columns if x[c].nunique() == 1]
		if drop_columns:
			print(f'Dropping these columns (1 unique value): {drop_columns}\n')
			x = x.drop(drop_columns, axis=1)

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

	x, y = x.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32).squeeze()

	if val_size == test_size == 0:
		if x_transform:
			if callable(x_transform):
				# x_transform is a custom Python func
				x = x_transform(x)
			else:
				# sklearn transform e.g. MinMaxScaler, StandardScaler
				x = x_transform.fit_transform(x)

		if output_as_tensor:
			x, y = torch.tensor(x).float(), torch.tensor(y).float()

		return x, y, features

	x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, train_size=train_size, random_state=1)
	if val_size == 0 or test_size == 0:
		if x_transform:
			if callable(x_transform):
				x_train = x_transform(x_train)
				x_tmp = x_transform(x_tmp)
			else:
				# Fit only on x_train
				x_train = x_transform.fit_transform(x_train)
				x_tmp = x_transform.transform(x_tmp)

		if output_as_tensor:
			x_train = torch.tensor(x_train).float()
			x_tmp = torch.tensor(x_tmp).float()
			y_train = torch.tensor(y_train).float()
			y_tmp = torch.tensor(y_tmp).float()

		return x_train, y_train, x_tmp, y_tmp, features

	val_size /= (val_size + test_size)
	x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, train_size=val_size, random_state=1)
	if x_transform:
		if callable(x_transform):
			x_train = x_transform(x_train)
			x_val = x_transform(x_val)
			x_test = x_transform(x_test)
		else:
			# Fit only on x_train
			x_train = x_transform.fit_transform(x_train)
			x_val = x_transform.transform(x_val)
			x_test = x_transform.transform(x_test)

	if output_as_tensor:
		x_train = torch.tensor(x_train).float()
		x_val = torch.tensor(x_val).float()
		x_test = torch.tensor(x_test).float()
		y_train = torch.tensor(y_train).float()
		y_val = torch.tensor(y_val).float()
		y_test = torch.tensor(y_test).float()

	return x_train, y_train, x_val, y_val, x_test, y_test, features
