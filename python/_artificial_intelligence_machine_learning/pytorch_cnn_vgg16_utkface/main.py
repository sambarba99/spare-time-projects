"""
VGG16-based PyTorch CNN for age prediction and gender/race classification of UTKFace

Author: Sam Barba
Created 30/10/2022
"""

from collections import Counter
from pathlib import Path

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
from _utils.plotting import plot_confusion_matrix, plot_roc_curve, plot_torch_model
from conv_net import CNN


np.random.seed(1)
pd.set_option('display.width', None)
pd.set_option('max_colwidth', None)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

DATASET_DICT = {
	'race': {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'other'},
	'gender': {0: 'male', 1: 'female'}
}
IMG_SIZE = 128
BATCH_SIZE = 32
BATCH_SIZE_TEST = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Weighting the losses according to how well the model generally does for each output
# (e.g. race classification is weakest, so its loss is given the most weight)
AGE_LOSS_WEIGHT = 0.1
GENDER_LOSS_WEIGHT = 4
RACE_LOSS_WEIGHT = 10


def create_data_loaders(df):
	x_path = df['img_path']
	y_age = df['age'].to_numpy()
	y_gender = df['gender_id'].to_numpy()
	y_race = df['race_id'].to_numpy()

	transform = transforms.Compose([
		transforms.Resize(IMG_SIZE),
		transforms.ToTensor()  # Scale to [0,1]
	])

	x = [
		transform(Image.open(img_path)) for img_path in
		tqdm(x_path, desc='Preprocessing images', unit='imgs', ascii=True)
	]
	y_age = torch.tensor(y_age).float()
	y_gender = torch.tensor(y_gender).float()
	y_race = torch.tensor(y_race).long()

	# Create train/validation/test sets (ratio 0.96:0.02:0.02), stratifying based on binned age + gender + race
	indices = np.arange(len(x))
	age_bins = np.linspace(0, 100, 9)
	age_bin_indices = np.digitize(y_age, age_bins)
	stratify_labels = [
		f'{age_bin_idx}_{int(gender)}_{race}' for age_bin_idx, gender, race
		in zip(age_bin_indices, y_gender, y_race)
	]
	strat_label_counts = Counter(stratify_labels)
	# Val and test are 0.02, so any label that occurs < 1/0.02 = 50 times is 'rare'
	stratify_labels = np.array([
		lbl if strat_label_counts[lbl] >= 50 else 'rare'
		for lbl in stratify_labels
	])
	train_idx, tmp_idx = train_test_split(indices, train_size=0.96, stratify=stratify_labels, random_state=1)
	val_idx, test_idx = train_test_split(tmp_idx, train_size=0.5, stratify=stratify_labels[tmp_idx], random_state=1)

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

	# Standardise images using training mean and std
	mean = torch.zeros(3)
	sq_mean = torch.zeros(3)
	num_pixels = IMG_SIZE * IMG_SIZE * len(x_train)
	for img in tqdm(x_train, desc='Calculating mean and std of x_train', unit='imgs', ascii=True):
		mean += img.sum(dim=[1, 2])
		sq_mean += (img ** 2).sum(dim=[1, 2])
	mean /= num_pixels
	sq_mean /= num_pixels
	std = (sq_mean - mean ** 2).sqrt()
	norm_transform = transforms.Normalize(mean.tolist(), std.tolist())
	for idx, img in enumerate(x_train):
		x_train[idx] = norm_transform(img)
	for idx, img in enumerate(x_val):
		x_val[idx] = norm_transform(img)
	for idx, img in enumerate(x_test):
		x_test[idx] = norm_transform(img)

	train_dataset = CustomDataset(x_train, y_train_age, y_train_gender, y_train_race)
	val_dataset = CustomDataset(x_val, y_val_age, y_val_gender, y_val_race)
	test_dataset = CustomDataset(x_test, y_test_age, y_test_gender, y_test_race)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST)

	return train_loader, val_loader, test_loader, mean, std


if __name__ == '__main__':
	# Convert data to dataframe

	data = []
	for p in Path('C:/Users/sam/Desktop/projects/datasets/utkface').glob('*.jpg'):
		y = str(p).split('\\')[-1]
		age, gender, race = y.split('_')[:3]
		data.append((str(p), int(age), int(gender), int(race)))

	df = pd.DataFrame(data, columns=['img_path', 'age', 'gender_id', 'race_id'])
	df['gender_label'] = df['gender_id'].map(DATASET_DICT['gender'])
	df['race_label'] = df['race_id'].map(DATASET_DICT['race'])

	print(f'\nRaw data:\n{df}\n')

	# Plot some examples

	rand_indices = np.random.choice(range(df.shape[0]), size=16, replace=False)
	_, axes = plt.subplots(nrows=4, ncols=4, figsize=(7, 7))
	plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.3)
	for idx, ax in zip(rand_indices, axes.flatten()):
		sample = Image.open(df['img_path'][idx])
		ax.imshow(sample)
		ax.axis('off')
		ax.set_title(f"{df['age'][idx]}, {df['gender_label'][idx]}, {df['race_label'][idx]}")
	plt.suptitle('Raw data samples (age, gender, race)', y=0.96)
	plt.show()

	# Plot output feature distributions

	fig, axes = plt.subplots(nrows=3, figsize=(8, 6))
	plt.subplots_adjust(hspace=0.4)
	for ax, col in zip(axes, ['age', 'gender_label', 'race_label']):
		if col == 'age':
			histogram_bins = list(range(0, 110, 10)) + [max(df[col])]
			ax.hist(df[col], bins=histogram_bins)
			ax.set_xticks(histogram_bins)
		else:
			unique_values_counts = df[col].value_counts()
			ax.bar(unique_values_counts.index, unique_values_counts.values)
		ax.set_xlabel(col.removesuffix('_label').capitalize())
	fig.supylabel('Count')
	plt.suptitle('Output feature distributions', y=0.94)
	plt.show()

	# Define data loaders and model

	train_loader, val_loader, test_loader, mean, std = create_data_loaders(df)

	model = CNN().to(DEVICE)
	print(f'\nModel:\n{model}')
	plot_torch_model(model, (3, IMG_SIZE, IMG_SIZE), device=DEVICE, out_file='./model_architecture')

	loss_func_age = torch.nn.MSELoss()
	loss_func_gender = torch.nn.BCEWithLogitsLoss()
	loss_func_race = torch.nn.CrossEntropyLoss()
	mae_func_age = torch.nn.L1Loss()

	if Path('./model.pth').exists():
		model.load_state_dict(torch.load('./model.pth', map_location=DEVICE))
	else:
		# Train model

		print('\n----- TRAINING -----\n')

		optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
		early_stopping = EarlyStopping(
			model=model, patience=10, mode='max',
			track_best_weights=False, print_precision_on_stop=2
		)
		history = {'age_val_MAE': [], 'gender_val_F1': [], 'race_val_F1': []}

		for epoch in range(1, NUM_EPOCHS + 1):
			progress_bar = tqdm(range(len(train_loader)), unit='batches', ascii=True)
			model.train()

			for x_train, y_train_age, y_train_gender, y_train_race in train_loader:
				progress_bar.update()
				progress_bar.set_description(f'Epoch {epoch}/{NUM_EPOCHS}')

				x_train = x_train.to(DEVICE)
				y_train_age = y_train_age.to(DEVICE)
				y_train_gender = y_train_gender.to(DEVICE)
				y_train_race = y_train_race.to(DEVICE)

				age_pred, gender_logits, race_logits = model(x_train)

				age_loss = loss_func_age(age_pred.squeeze(), y_train_age)
				gender_loss = loss_func_gender(gender_logits.squeeze(), y_train_gender)
				race_loss = loss_func_race(race_logits, y_train_race)

				# Apply loss weights
				weighted_loss = age_loss * AGE_LOSS_WEIGHT \
					+ gender_loss * GENDER_LOSS_WEIGHT \
					+ race_loss * RACE_LOSS_WEIGHT

				optimiser.zero_grad()
				weighted_loss.backward()
				optimiser.step()

				progress_bar.set_postfix_str(
					f'age_loss={age_loss.item():.2f}, '
					f'gender_loss={gender_loss.item():.4f}, '
					f'race_loss={race_loss.item():.4f}'
				)

			age_val_loss_total = age_val_mae_total = 0
			gender_val_loss_total = 0
			race_val_loss_total = 0
			all_gender_preds, all_gender_labels = [], []
			all_race_preds, all_race_labels = [], []

			model.eval()
			with torch.inference_mode():
				for x_val, y_val_age, y_val_gender, y_val_race in val_loader:
					x_val = x_val.to(DEVICE)
					y_val_age = y_val_age.to(DEVICE)
					y_val_gender = y_val_gender.to(DEVICE)
					y_val_race = y_val_race.to(DEVICE)

					age_val_pred, gender_val_logits, race_val_logits = model(x_val)
					age_val_pred = age_val_pred.squeeze()
					gender_val_logits = gender_val_logits.squeeze()

					gender_val_probs = torch.sigmoid(gender_val_logits)
					gender_val_pred = gender_val_probs.round()
					race_val_pred = race_val_logits.argmax(dim=1)

					all_gender_preds.append(gender_val_pred.cpu())
					all_gender_labels.append(y_val_gender.cpu())
					all_race_preds.append(race_val_pred.cpu())
					all_race_labels.append(y_val_race.cpu())

					age_val_loss_total += loss_func_age(age_val_pred, y_val_age).item()
					age_val_mae_total += mae_func_age(age_val_pred, y_val_age).item()
					gender_val_loss_total += loss_func_gender(gender_val_logits, y_val_gender).item()
					race_val_loss_total += loss_func_race(race_val_logits, y_val_race).item()

			age_val_loss = age_val_loss_total / len(val_loader)
			age_val_mae = age_val_mae_total / len(val_loader)
			gender_val_loss = gender_val_loss_total / len(val_loader)
			gender_val_f1 = f1_score(torch.cat(all_gender_labels), torch.cat(all_gender_preds))
			race_val_loss = race_val_loss_total / len(val_loader)
			race_val_f1 = f1_score(torch.cat(all_race_labels), torch.cat(all_race_preds), average='weighted')

			progress_bar.set_postfix_str(
				f'age_val_loss={age_val_loss:.2f}, '
				f'age_val_MAE={age_val_mae:.2f}, '
				f'gender_val_loss={gender_val_loss:.4f}, '
				f'gender_val_F1={gender_val_f1:.4f}, '
				f'race_val_loss={race_val_loss:.4f}, '
				f'race_val_F1={race_val_f1:.4f}'
			)
			progress_bar.close()

			torch.save(
				model.state_dict(),
				f'./model_{age_val_mae:.2f}_{gender_val_f1:.2f}_{race_val_f1:.2f}.pth'
			)

			history['age_val_MAE'].append(age_val_mae)
			history['gender_val_F1'].append(gender_val_f1)
			history['race_val_F1'].append(race_val_f1)

			# Condition early stopping on race val F1 score, as this is the weakest model output
			if early_stopping(race_val_f1):
				break

		# Plot training metrics
		_, (ax_age_val_mae, ax_gender_f1, ax_race_f1) = plt.subplots(nrows=3, sharex=True, figsize=(8, 5))
		plt.subplots_adjust(hspace=0.4)
		ax_age_val_mae.plot(history['age_val_MAE'])
		ax_gender_f1.plot(history['gender_val_F1'])
		ax_race_f1.plot(history['race_val_F1'])
		ax_age_val_mae.set_ylabel('MAE')
		ax_gender_f1.set_ylabel('F1')
		ax_race_f1.set_ylabel('F1')
		ax_age_val_mae.set_title('Age val MAE', fontsize=11)
		ax_gender_f1.set_title('Gender val F1 score', fontsize=11)
		ax_race_f1.set_title('Race val F1 score', fontsize=11)
		ax_race_f1.set_xlabel('Epoch')
		plt.suptitle('Training metrics', x=0.505)
		plt.show()

	# Test model

	print('\n----- TESTING -----\n')

	age_test_loss_total = age_test_mae_total = 0
	gender_test_loss_total = 0
	race_test_loss_total = 0
	all_gender_preds, all_gender_labels, all_gender_logits = [], [], []
	all_race_preds, all_race_labels, all_race_logits = [], [], []

	pil_image_transform = transforms.Compose([
		transforms.Lambda(lambda img: img * std.view(3, 1, 1) + mean.view(3, 1, 1)),  # De-standardise
		transforms.ToPILImage()
	])

	model.eval()
	with torch.inference_mode():
		for test_idx, (x_test, y_test_age, y_test_gender, y_test_race) in enumerate(test_loader):
			x_test = x_test.to(DEVICE)
			y_test_age = y_test_age.to(DEVICE)
			y_test_gender = y_test_gender.to(DEVICE)
			y_test_race = y_test_race.to(DEVICE)

			age_test_pred, gender_test_logits, race_test_logits = model(x_test)
			age_test_pred = age_test_pred.squeeze()
			gender_test_logits = gender_test_logits.squeeze()

			gender_test_probs = torch.sigmoid(gender_test_logits)
			gender_test_pred = gender_test_probs.round()
			race_test_pred = race_test_logits.argmax(dim=1)

			all_gender_preds.append(gender_test_pred.cpu())
			all_gender_labels.append(y_test_gender.cpu())
			all_gender_logits.append(gender_test_logits.cpu())
			all_race_preds.append(race_test_pred.cpu())
			all_race_labels.append(y_test_race.cpu())
			all_race_logits.append(race_test_logits.cpu())

			age_test_loss_total += loss_func_age(age_test_pred, y_test_age).item()
			age_test_mae_total += mae_func_age(age_test_pred, y_test_age).item()
			gender_test_loss_total += loss_func_gender(gender_test_logits, y_test_gender).item()
			race_test_loss_total += loss_func_race(race_test_logits, y_test_race).item()

			if len(x_test) < BATCH_SIZE_TEST:  # Last batch may be smaller than BATCH_SIZE_TEST
				break

			# Plot test images with model outputs

			_, axes = plt.subplots(nrows=4, ncols=4, figsize=(9, 8))
			plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05, hspace=0.45)

			for idx, ax in enumerate(axes.flatten()):
				img, y_age, y_gender, y_race = x_test[idx], y_test_age[idx], y_test_gender[idx], y_test_race[idx]

				y_gender_label = DATASET_DICT['gender'][int(y_gender.item())]
				y_race_label = DATASET_DICT['race'][y_race.item()]

				age_pred = round(age_test_pred[idx].item())
				gender_pred = int(gender_test_pred[idx].item())
				race_pred = race_test_pred[idx].item()
				gender_pred_label = DATASET_DICT['gender'][gender_pred]
				race_pred_label = DATASET_DICT['race'][race_pred]

				ax.imshow(pil_image_transform(img.cpu()))
				ax.axis('off')
				ax.set_title(
					f'Pred: {age_pred}, {gender_pred_label}, {race_pred_label}'
					f'\nActual: {int(y_age)}, {y_gender_label}, {y_race_label}',
					fontsize=10
				)
			plt.suptitle('Test output examples', y=0.95)
			plt.savefig(f'./test_output{test_idx}.png')
			plt.close()

	age_test_loss = age_test_loss_total / len(test_loader)
	age_test_mae = age_test_mae_total / len(test_loader)
	gender_test_loss = gender_test_loss_total / len(test_loader)
	gender_test_f1 = f1_score(torch.cat(all_gender_labels), torch.cat(all_gender_preds))
	race_test_loss = race_test_loss_total / len(test_loader)
	race_test_f1 = f1_score(torch.cat(all_race_labels), torch.cat(all_race_preds), average='weighted')

	print(f'Age test loss: {age_test_loss:.2f}')
	print(f'Age test MAE: {age_test_mae:.2f}')
	print(f'Gender test loss: {gender_test_loss:.4f}')
	print(f'Gender test F1: {gender_test_f1:.4f}')
	print(f'Race test loss: {race_test_loss:.4f}')
	print(f'Race test F1: {race_test_f1:.4f}')

	# ROC curve for gender output
	gender_test_probs = torch.sigmoid(torch.cat(all_gender_logits))
	plot_roc_curve(torch.cat(all_gender_labels), gender_test_probs, 'Test ROC curve for gender classification')

	# Confusion matrix for gender output
	gender_test_pred = gender_test_probs.round()
	plot_confusion_matrix(
		torch.cat(all_gender_labels),
		gender_test_pred,
		sorted(DATASET_DICT['gender'].values()),
		'Test confusion matrix for gender classification'
	)

	# Confusion matrix for race output
	race_test_pred = torch.cat(all_race_logits).argmax(dim=1)
	plot_confusion_matrix(
		torch.cat(all_race_labels),
		race_test_pred,
		sorted(DATASET_DICT['race'].values()),
		'Test confusion matrix for race classification'
	)
