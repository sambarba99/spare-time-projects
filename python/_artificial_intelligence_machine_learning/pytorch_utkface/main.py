"""
PyTorch VGG16-based CNN for age prediction and gender/race classification of UTKFace dataset

Author: Sam Barba
Created 30/10/2022
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from _utils.model_architecture_plots import plot_model
from _utils.model_evaluation_plots import plot_confusion_matrix, plot_roc_curve
from conv_net import CNN


np.random.seed(1)
pd.set_option('display.width', None)
pd.set_option('max_colwidth', None)
torch.manual_seed(1)

DATA_PATH = 'C:/Users/Sam/Desktop/projects/datasets/utkface'  # https://www.kaggle.com/datasets/jangedoo/utkface-new
DATASET_DICT = {
	'race_id': {'0': 'white', '1': 'black', '2': 'asian', '3': 'indian', '4': 'other'},
	'gender_id': {'0': 'male', '1': 'female'}
}
INPUT_SIZE = 128
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

# Weighting the losses according to how well the model generally does for each output
# (e.g. race classification is weakest, so its loss weighs the most)
AGE_LOSS_WEIGHT = 0.1
GENDER_LOSS_WEIGHT = 4
RACE_LOSS_WEIGHT = 10


def create_data_loaders(df):
	x_path = df['img_path'].to_numpy()
	y_age = df['age'].to_numpy()
	y_gender = pd.get_dummies(df['gender'], prefix='gender', drop_first=True, dtype=int).to_numpy().squeeze()
	y_race = pd.get_dummies(df['race'], prefix='race', dtype=int).to_numpy()

	# Preprocess images now instead of during training (faster pipeline overall)

	transform = transforms.Compose([
		transforms.Resize(INPUT_SIZE),
		transforms.ToTensor()  # Automatically normalises to [0,1]
	])

	x = [
		transform(Image.open(fp)) for fp in
		tqdm(x_path, desc='Preprocessing images', unit='imgs', ascii=True)
	]
	y_age = torch.tensor(y_age).float()
	y_gender = torch.tensor(y_gender).float()
	y_race = torch.tensor(y_race).float()

	# Create train/validation/test sets (ratio 0.96:0.02:0.02)
	# Stratify based on binned age + gender + race

	indices = np.arange(len(x))
	age_bins = np.linspace(0, 100, 9)
	age_bin_indices = np.digitize(y_age, age_bins)
	stratify_labels = np.array([
		f'{age_bin_idx}_{gender}_{race}' for age_bin_idx, gender, race
		in zip(age_bin_indices, df['gender'], df['race'])
	])
	train_val_idx, test_idx = train_test_split(indices, train_size=0.98, stratify=stratify_labels, random_state=1)
	train_idx, val_idx = train_test_split(train_val_idx, train_size=0.98, stratify=stratify_labels[train_val_idx], random_state=1)

	x_train = [x[i] for i in train_idx]
	x_val = [x[i] for i in val_idx]
	x_test = [x[i] for i in test_idx]
	y_train_age = y_age[train_idx]
	y_val_age = y_age[val_idx]
	y_test_age = y_age[test_idx]
	y_train_gender = y_gender[train_idx]
	y_val_gender = y_gender[val_idx]
	y_test_gender = y_gender[test_idx]
	y_train_race = y_race[train_idx]
	y_val_race = y_race[val_idx]
	y_test_race = y_race[test_idx]

	train_dataset = CustomDataset(x_train, y_train_age, y_train_gender, y_train_race)
	val_dataset = CustomDataset(x_val, y_val_age, y_val_gender, y_val_race)
	test_dataset = CustomDataset(x_test, y_test_age, y_test_gender, y_test_race)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
	val_loader = DataLoader(val_dataset, batch_size=len(x_val))
	test_loader = DataLoader(test_dataset, batch_size=len(x_test))

	return train_loader, val_loader, test_loader


if __name__ == '__main__':
	# 1. Convert data to dataframe

	data = []
	for img_path in os.listdir(DATA_PATH):
		age, gender, race = img_path.split('_')[:3]
		data.append((
			f'{DATA_PATH}/{img_path}',
			int(age),
			DATASET_DICT['gender_id'][gender],
			DATASET_DICT['race_id'][race]
		))

	df = pd.DataFrame(data, columns=['img_path', 'age', 'gender', 'race'])
	print(f'\nRaw data:\n{df}\n')

	# 2. Plot some examples

	rand_indices = np.random.choice(range(df.shape[0]), size=16, replace=False)
	_, axes = plt.subplots(nrows=4, ncols=4, figsize=(7, 7))
	plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.3)
	for idx, ax in zip(rand_indices, axes.flatten()):
		sample = Image.open(df['img_path'][idx])
		ax.imshow(sample)
		ax.axis('off')
		ax.set_title(f"{df['age'][idx]}, {df['gender'][idx]}, {df['race'][idx]}")
	plt.suptitle('Data samples (age, gender, race)', y=0.96)
	plt.show()

	# 3. Plot output feature distributions

	fig, axes = plt.subplots(nrows=3, figsize=(8, 6))
	plt.subplots_adjust(hspace=0.4)
	for ax, col in zip(axes, ['age', 'gender', 'race']):
		if col == 'age':
			histogram_bins = list(range(0, 110, 10)) + [max(df[col])]
			ax.hist(df[col], bins=histogram_bins)
			ax.set_xticks(histogram_bins)
		else:
			unique_values_counts = df[col].value_counts()
			ax.bar(unique_values_counts.index, unique_values_counts.values)
		ax.set_xlabel(col.capitalize())
	fig.supylabel('Count')
	plt.suptitle('Output feature distributions', y=0.94)
	plt.show()

	# 4. Define data loaders and model

	train_loader, val_loader, test_loader = create_data_loaders(df)

	model = CNN()
	model.to('cpu')
	print(f'Model:\n{model}')
	plot_model(model, (3, INPUT_SIZE, INPUT_SIZE))

	loss_func_age = torch.nn.MSELoss()
	loss_func_gender = torch.nn.BCELoss()
	loss_func_race = torch.nn.CrossEntropyLoss()
	metric_age = torch.nn.L1Loss()  # MAE

	if os.path.exists('./model.pth'):
		model.load_state_dict(torch.load('./model.pth'))
	else:
		# 5. Train model

		print('\n----- TRAINING -----\n')

		optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
		early_stopping = EarlyStopping(patience=10, min_delta=0, mode='max')
		history = {'age_val_MAE': [], 'gender_val_F1': [], 'race_val_F1': []}

		for epoch in range(1, NUM_EPOCHS + 1):
			progress_bar = tqdm(range(len(train_loader)), unit='batches', ascii=True)
			model.train()

			for x_train, y_train_age, y_train_gender, y_train_race in train_loader:
				progress_bar.update()
				progress_bar.set_description(f'Epoch {epoch}/{NUM_EPOCHS}')
				age_pred, gender_pred_probs, race_pred_probs = model(x_train)

				age_loss = loss_func_age(age_pred, y_train_age)
				age_mae = metric_age(age_pred, y_train_age).item()
				gender_loss = loss_func_gender(gender_pred_probs, y_train_gender)
				gender_f1 = f1_score(y_train_gender, gender_pred_probs.detach().numpy().round())
				race_loss = loss_func_race(race_pred_probs, y_train_race)
				race_f1 = f1_score(y_train_race.argmax(dim=1), race_pred_probs.argmax(dim=1), average='weighted')

				# Apply loss weights
				weighted_loss = age_loss * AGE_LOSS_WEIGHT + gender_loss * GENDER_LOSS_WEIGHT + race_loss * RACE_LOSS_WEIGHT

				optimiser.zero_grad()
				weighted_loss.backward()
				optimiser.step()

				progress_bar.set_postfix_str(
					f'age_loss={age_loss.item():.4f}, '
					f'age_MAE={age_mae:.4f}, '
					f'gender_loss={gender_loss.item():.4f}, '
					f'gender_F1={gender_f1:.4f}, '
					f'race_loss={race_loss:.4f}, '
					f'race_F1={race_f1:.4f}'
				)

			x_val, y_val_age, y_val_gender, y_val_race = next(iter(val_loader))
			model.eval()
			with torch.inference_mode():
				age_val_pred, gender_val_pred_probs, race_val_pred_probs = model(x_val)

			age_val_loss = loss_func_age(age_val_pred, y_val_age).item()
			age_val_mae = metric_age(age_val_pred, y_val_age).item()
			gender_val_loss = loss_func_gender(gender_val_pred_probs, y_val_gender).item()
			gender_val_f1 = f1_score(y_val_gender, gender_val_pred_probs.detach().numpy().round())
			race_val_loss = loss_func_race(race_val_pred_probs, y_val_race).item()
			race_val_f1 = f1_score(y_val_race.argmax(dim=1), race_val_pred_probs.argmax(dim=1), average='weighted')
			progress_bar.set_postfix_str(
				f'{progress_bar.postfix}, '
				f'age_val_loss={age_val_loss:.4f}, '
				f'age_val_MAE={age_val_mae:.4f}, '
				f'gender_val_loss={gender_val_loss:.4f}, '
				f'gender_val_F1={gender_val_f1:.4f}, '
				f'race_val_loss={race_val_loss:.4f}, '
				f'race_val_F1={race_val_f1:.4f}'
			)
			progress_bar.close()

			model_path = f'./model_{age_val_mae:.2f}_{gender_val_f1:.2f}_{race_val_f1:.2f}.pth'
			torch.save(model.state_dict(), model_path)  # Best: model_5.72_0.90_0.78.pth

			history['age_val_MAE'].append(age_val_mae)
			history['gender_val_F1'].append(gender_val_f1)
			history['race_val_F1'].append(race_val_f1)

			# Condition early stopping on race val F1 score, as this is the least accurate model output
			if early_stopping(race_val_f1, None):
				print('Early stopping at epoch', epoch)
				break

		# Plot training metrics
		_, (ax_age_val_mae, ax_gender_f1, ax_race_f1) = plt.subplots(nrows=3, sharex=True, figsize=(8, 5))
		ax_age_val_mae.plot(history['age_val_MAE'])
		ax_gender_f1.plot(history['gender_val_F1'])
		ax_race_f1.plot(history['race_val_F1'])
		ax_age_val_mae.set_ylabel('MAE')
		ax_gender_f1.set_ylabel('F1')
		ax_race_f1.set_ylabel('F1')
		ax_age_val_mae.set_title('Age val MAE')
		ax_gender_f1.set_title('Gender val F1 score')
		ax_race_f1.set_title('Race val F1 score')
		ax_race_f1.set_xlabel('Epoch')
		plt.suptitle('Training metrics')
		plt.show()

	# 6. Testing

	print('\n----- TESTING -----\n')

	x_test, y_test_age, y_test_gender, y_test_race = next(iter(test_loader))
	model.eval()
	with torch.inference_mode():
		age_test_pred, gender_test_pred_probs, race_test_pred_probs = model(x_test)

	print('Test age MAE:', metric_age(age_test_pred, y_test_age).item())

	# ROC curve for gender output
	plot_roc_curve(y_test_gender, gender_test_pred_probs, 'ROC curve for gender classification')

	# Confusion matrices for gender and race outputs
	gender_test_pred_labels = gender_test_pred_probs.round()
	race_test_pred_labels = race_test_pred_probs.argmax(dim=1)
	f1_gender = f1_score(y_test_gender, gender_test_pred_labels)
	f1_race = f1_score(y_test_race.argmax(dim=1), race_test_pred_labels, average='weighted')
	plot_confusion_matrix(
		y_test_gender,
		gender_test_pred_labels,
		sorted(DATASET_DICT['gender_id'].values()),
		f'Test confusion matrix for gender classification\n(F1 score: {f1_gender:.3f})'
	)
	plot_confusion_matrix(
		y_test_race.argmax(dim=1),
		race_test_pred_labels,
		sorted(DATASET_DICT['race_id'].values()),
		f'Test confusion matrix for race classification\n(F1 score: {f1_race:.3f})'
	)

	# Plot first 16 test set images with outputs

	_, axes = plt.subplots(nrows=4, ncols=4, figsize=(9, 8))
	plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05, hspace=0.45)

	for idx, ax in enumerate(axes.flatten()):
		img, y_age, y_gender, y_race = x_test[idx], y_test_age[idx], y_test_gender[idx], y_test_race[idx]

		# When doing get_dummies on the gender column in the data preparation step,
		# 'male' is assigned 1 (but it is 0 in DATASET_DICT) so we must subtract it from 1.
		y_gender = 1 - int(y_gender)
		y_gender_label = DATASET_DICT['gender_id'][str(y_gender)]
		y_race = y_race.argmax().item()
		y_race_label = DATASET_DICT['race_id'][str(y_race)]

		age_pred = round(age_test_pred[idx].item())
		gender_pred = 1 - round(gender_test_pred_probs[idx].item())
		gender_pred_label = DATASET_DICT['gender_id'][str(gender_pred)]
		race_pred = race_test_pred_probs[idx].argmax().item()
		race_pred_label = DATASET_DICT['race_id'][str(race_pred)]

		img = (img * 255).type(torch.uint8).permute(1, 2, 0)
		ax.imshow(img)
		ax.axis('off')
		ax.set_title(
			f'Pred: {age_pred}, {gender_pred_label}, {race_pred_label}'
			+ f'\nActual: {int(y_age)}, {y_gender_label}, {y_race_label}',
			fontsize=11
		)
	plt.suptitle('Test output examples', y=0.96)
	plt.show()
