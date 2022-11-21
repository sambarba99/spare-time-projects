"""
(PyTorch) VGG16-based CNN for age prediction and gender/race classification of UTKFace dataset

Author: Sam Barba
Created 30/10/2022
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score
import torch
from torch import nn
from tqdm import tqdm

from conv_net import CNN
from early_stopping import EarlyStopping

DATA_PATH = r'C:\Users\Sam Barba\Desktop\Programs\datasets\UTKFace'  # Available from Kaggle
DATASET_DICT = {
	'race_id': {'0': 'white', '1': 'black', '2': 'asian', '3': 'indian', '4': 'other'},
	'gender_id': {'0': 'male', '1': 'female'}
}
BATCH_SIZE = 32

plt.rcParams['figure.figsize'] = (9, 6)
pd.set_option('display.width', None)
pd.set_option('max_colwidth', None)

def clean_data(df):
	gender, race = df.pop('gender'), df.pop('race')

	gender = pd.get_dummies(gender, prefix='gender', drop_first=True)
	race_one_hot = pd.get_dummies(race, prefix='race')

	df = pd.concat([df, gender, race_one_hot], axis=1)
	return df

def preprocess_img(path):
	img = Image.open(path)
	img = img.resize((128, 128))
	img = np.array(img) / 255  # Scale from 0-1
	img = img.reshape(3, 128, 128)  # In PyTorch, colour channels come first

	return img

def create_splits(df_shape):
	train_prop, val_prop = 0.8, 0.1  # Test proportion = 0.1
	train_size = int(df_shape * train_prop)
	val_size = int(df_shape * val_prop)

	perm = np.random.permutation(df_shape)
	train_idx = perm[:train_size]
	val_idx = perm[train_size:train_size + val_size]
	test_idx = perm[train_size + val_size:]

	return train_idx, val_idx, test_idx

def data_generator(*, preprocessed_images, df, idx, device):
	batches = [idx[i:i + BATCH_SIZE] for i in range(0, len(idx), BATCH_SIZE)]

	for batch in batches:
		images = [preprocessed_images[i] for i in batch]
		ages = [df.iloc[i, 1] for i in batch]
		genders = [df.iloc[i, 2] for i in batch]
		races = [df.iloc[i, 3:] for i in batch]

		batch_x = torch.from_numpy(np.array(images)).float().to(device)
		batch_y = [torch.from_numpy(np.array(ages)).float().to(device),
			torch.from_numpy(np.array(genders)).float().to(device),
			torch.from_numpy(np.array(races).astype(float)).float().to(device)]

		yield batch_x, batch_y

if __name__ == '__main__':
	np.random.seed(1)
	torch.manual_seed(1)

	# 1. Convert data to dataframe

	data = []
	for img_path in os.listdir(DATA_PATH):
		age, gender, race = img_path.split('_')[:3]
		age = int(age)
		gender = DATASET_DICT['gender_id'][gender]
		race = DATASET_DICT['race_id'][race]
		data.append((fr'{DATA_PATH}\{img_path}', age, gender, race))

	df = pd.DataFrame(data, columns=['img_path', 'age', 'gender', 'race'])

	# 2. Plot some examples

	rand_idx = np.random.choice(range(df.shape[0]), size=9, replace=False)
	_, axes = plt.subplots(nrows=3, ncols=3)
	plt.subplots_adjust(top=0.85, bottom=0.05, hspace=0.3, wspace=0)
	for idx, ax in enumerate(axes.flatten()):
		r = rand_idx[idx]
		sample = Image.open(df['img_path'][r])
		ax.imshow(sample)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title(f'{df["age"][r]}, {df["gender"][r]}, {df["race"][r]}')
	plt.suptitle('Data examples (age, gender, race)', x=0.508, y=0.95)
	plt.show()

	# 3. Plot output feature distributions

	fig, axes = plt.subplots(nrows=3)
	plt.subplots_adjust(hspace=0.4)
	for ax, col in zip(axes, ['age', 'gender', 'race']):
		unique_values, counts = np.unique(df[col].to_numpy(), return_counts=True)
		ax.bar(unique_values, counts)
		ax.set_xlabel(col.capitalize())
	fig.supylabel('Count')
	plt.suptitle('Output feature distributions', y=0.94)
	plt.show()

	# 4. Clean up data and create splits for data generators

	print(f'\nRaw data:\n{df}')
	df = clean_data(df)
	print(f'\nCleaned data:\n{df}\n')

	processed = [
		preprocess_img(p) for p in
		tqdm(df['img_path'], desc='Preprocessing images', ascii=True)
	]

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	loss_func_age = nn.MSELoss()
	loss_func_gender = nn.BCELoss()
	loss_func_race = nn.CrossEntropyLoss()
	metric_age = nn.L1Loss()

	train_idx, val_idx, test_idx = create_splits(df.shape[0])

	choice = input('\nEnter T to train a new model or L to load existing one\n>>> ').upper()

	if choice == 'T':
		# 5. Build model

		model = CNN().to(device)
		optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
		print(f'\nModel:\n{model}')

		# 6. Train model

		print('\n----- TRAINING -----\n')

		n_train_batches = int(np.ceil(len(train_idx) / BATCH_SIZE))
		n_val_batches = int(np.ceil(len(val_idx) / BATCH_SIZE))

		early_stopping = EarlyStopping(patience=2, min_delta=0)

		for epoch in range(20):
			model.train()
			train_gen = data_generator(preprocessed_images=processed, df=df, idx=train_idx, device=device)

			for train_batch in tqdm(train_gen, desc=f'Epoch: {epoch} | batch', total=n_train_batches, ascii=True):
				x, y = train_batch
				y_pred = model(x)
				age_pred, gender_probs, race_probs = map(torch.squeeze, y_pred)

				age_loss = loss_func_age(age_pred, y[0])
				gender_loss = loss_func_gender(gender_probs, y[1])
				race_loss = loss_func_race(race_probs, y[2])
				loss = age_loss * 0.1 + gender_loss * 10 + race_loss * 4  # Apply loss weights

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

			model.eval()
			val_gen = data_generator(preprocessed_images=processed, df=df, idx=val_idx, device=device)
			mean_age_val_mae = mean_gender_val_f1 = mean_race_val_f1 = 0
			mean_val_loss = 0

			with torch.inference_mode():
				for val_batch in val_gen:
					x, y = val_batch
					y_val_pred = model(x)
					age_val_pred, gender_val_probs, race_val_probs = map(torch.squeeze, y_val_pred)

					age_val_loss = loss_func_age(age_val_pred, y[0])
					gender_val_loss = loss_func_gender(gender_val_probs, y[1])
					race_val_loss = loss_func_race(race_val_probs, y[2])
					mean_val_loss += (age_val_loss * 0.1 + gender_val_loss * 10 + race_val_loss * 4) / n_val_batches

					age_val_mae = metric_age(age_val_pred, y[0])
					gender_val_f1 = f1_score(y[1], gender_val_probs.round().detach().numpy())
					race_val_f1 = f1_score(y[2].argmax(dim=1), race_val_probs.argmax(dim=1), average='weighted')
					mean_age_val_mae += age_val_mae.item() / n_val_batches
					mean_gender_val_f1 += gender_val_f1 / n_val_batches
					mean_race_val_f1 += race_val_f1 / n_val_batches

			print(f'Epoch: {epoch} | Age val MAE: {mean_age_val_mae} | Gender val F1: {mean_gender_val_f1} | Race val F1: {mean_race_val_f1}')

			if early_stopping.check_stop(mean_val_loss, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights

		torch.save(model, 'model.pth')
	elif choice == 'L':
		model = torch.load('model.pth')
	else:
		raise ValueError('Bad choice')

	# 7. Testing/evaluation

	print('\n----- TESTING/EVALUATION -----\n')

	model.eval()
	test_gen = data_generator(preprocessed_images=processed, df=df, idx=test_idx, device=device)
	n_test_batches = int(np.ceil(len(test_idx) / BATCH_SIZE))
	mean_age_test_loss = mean_gender_test_loss = mean_race_test_loss = 0
	mean_age_test_mae = mean_gender_test_f1 = mean_race_test_f1 = 0

	with torch.inference_mode():
		for test_batch in test_gen:
			x, y = test_batch
			y_test_pred = model(x)
			age_test_pred, gender_test_probs, race_test_probs = map(torch.squeeze, y_test_pred)

			age_test_loss = loss_func_age(age_test_pred, y[0])
			gender_test_loss = loss_func_gender(gender_test_probs, y[1])
			race_test_loss = loss_func_race(race_test_probs, y[2])
			age_test_mae = metric_age(age_test_pred, y[0])
			gender_test_f1 = f1_score(y[1], gender_test_probs.round().detach().numpy())
			race_test_f1 = f1_score(y[2].argmax(dim=1), race_test_probs.argmax(dim=1), average='weighted')
			mean_age_test_loss += age_test_loss.item() / n_test_batches
			mean_gender_test_loss += gender_test_loss.item() / n_test_batches
			mean_race_test_loss += race_test_loss.item() / n_test_batches
			mean_age_test_mae += age_test_mae.item() / n_test_batches
			mean_gender_test_f1 += gender_test_f1 / n_test_batches
			mean_race_test_f1 += race_test_f1 / n_test_batches

	print('Test age loss (MSE):', mean_age_test_loss)
	print('Test gender loss (binary crossentropy):', mean_gender_test_loss)
	print('Test race loss (categorical crossentropy):', mean_race_test_loss)
	print('Test age MAE:', mean_age_test_mae)
	print('Test gender F1 score:', mean_gender_test_f1)
	print('Test race F1 score:', mean_race_test_f1)

	# Plot predictions of first 9 images of first test batch
	test_gen = data_generator(preprocessed_images=processed, df=df, idx=test_idx, device=device)
	first_batch = next(test_gen)
	images, (ages, genders, races) = first_batch

	_, axes = plt.subplots(nrows=3, ncols=3)
	plt.subplots_adjust(top=0.8, bottom=0.05, hspace=0.5, wspace=0.4)

	with torch.inference_mode():
		for idx, ax in enumerate(axes.flatten()):
			img, age, gender, race = images[idx], ages[idx], genders[idx], races[idx]

			# Subtract gender from 1, as model is predicting 'gender_male'.
			# If 'gender_male' = 1, the DATASET_DICT index should be 1 - 1 = 0
			# (it's 0 for male, 1 for female)
			gender = 1 - int(gender.item())
			gender = DATASET_DICT['gender_id'][str(gender)]
			race = race.argmax().item()
			race = DATASET_DICT['race_id'][str(race)]

			y_pred = model(img.unsqueeze(0))
			age_pred, gender_probs, race_probs = map(torch.squeeze, y_pred)
			age_pred = round(age_pred.item())
			gender_pred = 1 - round(gender_probs.item())
			gender_pred = DATASET_DICT['gender_id'][str(gender_pred)]
			race_pred = race_probs.argmax().item()
			race_pred = DATASET_DICT['race_id'][str(race_pred)]

			img = (img.numpy() * 255).astype(np.uint8).reshape((128, 128, 3))
			ax.imshow(Image.fromarray(img))
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_title(f'Predicted: {age_pred}, {gender_pred}, {race_pred}'
				+ f'\nActual: {int(age)}, {gender}, {race}')

		plt.suptitle('Test output examples', x=0.508, y=0.95)
		plt.savefig('output_examples.png')
		plt.show()
