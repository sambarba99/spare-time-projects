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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from conv_net import CNN
from custom_dataset import CustomDataset
from early_stopping import EarlyStopping


plt.rcParams['figure.figsize'] = (9, 6)
np.random.seed(1)
pd.set_option('display.width', None)
pd.set_option('max_colwidth', None)
torch.manual_seed(1)

DATA_PATH = 'C:/Users/Sam/Desktop/Projects/datasets/UTKFace'  # Available from Kaggle
DATASET_DICT = {
	'race_id': {'0': 'white', '1': 'black', '2': 'asian', '3': 'indian', '4': 'other'},
	'gender_id': {'0': 'male', '1': 'female'}
}
INPUT_SIZE = 128
N_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Weighting the losses according to how well the model generally does for each output
# (e.g. race classification is weakest, so its loss weighs the most)
AGE_LOSS_WEIGHT = 0.1
GENDER_LOSS_WEIGHT = 4
RACE_LOSS_WEIGHT = 10


def create_data_loaders(df):
	x_path = df['img_path'].to_numpy()
	y_age = df['age'].to_numpy()
	y_gender = pd.get_dummies(df['gender'], prefix='gender', drop_first=True).to_numpy().squeeze()
	y_race = pd.get_dummies(df['race'], prefix='race').to_numpy()

	# Shuffle
	idx_permutation = np.random.permutation(df.shape[0])
	x_path = x_path[idx_permutation]
	y_age = y_age[idx_permutation]
	y_gender = y_gender[idx_permutation]
	y_race = y_race[idx_permutation]

	# Pre-process images now instead of during training (faster pipeline overall)

	transform = transforms.Compose([
		transforms.Resize(INPUT_SIZE),
		# transforms.CenterCrop(INPUT_SIZE),
		transforms.ToTensor()  # Automatically normalises to [0,1]
	])

	x = [
		transform(Image.open(fp)) for fp in
		tqdm(x_path, desc='Preprocessing images', ascii=True)
	]

	# Split into train, validation, test sets (ratio 0.96:0.02:0.02)

	train_size = round(df.shape[0] * 0.96)
	val_size = round(df.shape[0] * 0.02)

	x_train = x[:train_size]
	x_val = x[train_size:train_size + val_size]
	x_test = x[train_size + val_size:]
	y_train_age = y_age[:train_size]
	y_val_age = y_age[train_size:train_size + val_size]
	y_test_age = y_age[train_size + val_size:]
	y_train_gender = y_gender[:train_size]
	y_val_gender = y_gender[train_size:train_size + val_size]
	y_test_gender = y_gender[train_size + val_size:]
	y_train_race = y_race[:train_size]
	y_val_race = y_race[train_size:train_size + val_size]
	y_test_race = y_race[train_size + val_size:]

	train_dataset = CustomDataset(x_train, y_train_age, y_train_gender, y_train_race)
	val_dataset = CustomDataset(x_val, y_val_age, y_val_gender, y_val_race)
	test_dataset = CustomDataset(x_test, y_test_age, y_test_gender, y_test_race)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
	val_loader = DataLoader(val_dataset, batch_size=len(x_val), shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=len(x_test), shuffle=False)

	return train_loader, val_loader, test_loader


def plot_confusion_matrix(y_name, actual, predictions, labels):
	cm = confusion_matrix(actual, predictions)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
	f1 = f1_score(actual, predictions, average='binary' if len(labels) == 2 else 'weighted')

	disp.plot(cmap='plasma')
	plt.title(f'Test confusion matrix for {y_name} classification\n(F1 score: {f1})')
	plt.show()


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

	rand_indices = np.random.choice(range(df.shape[0]), size=9, replace=False)
	_, axes = plt.subplots(nrows=3, ncols=3)
	plt.subplots_adjust(top=0.85, bottom=0.05, hspace=0.3, wspace=0)
	for idx, ax in zip(rand_indices, axes.flatten()):
		sample = Image.open(df['img_path'][idx])
		ax.imshow(sample)
		ax.axis('off')
		ax.set_title(f"{df['age'][idx]}, {df['gender'][idx]}, {df['race'][idx]}")
	plt.suptitle('Data samples (age, gender, race)', x=0.508, y=0.95)
	plt.show()

	# 3. Plot output feature distributions

	fig, axes = plt.subplots(nrows=3)
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
	print(f'Model:\n{model}')

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
		early_stopping = EarlyStopping(patience=5, min_delta=0, mode='max')
		history = {'age_val_MAE': [], 'gender_val_F1': [], 'race_val_F1': []}

		for epoch in range(1, N_EPOCHS + 1):
			model.train()
			for x_train, y_train_age, y_train_gender, y_train_race in train_loader:
				y_train_pred = model(x_train)
				age_pred, gender_pred_probs, race_pred_probs = map(torch.squeeze, y_train_pred)

				age_loss = loss_func_age(age_pred, y_train_age)
				gender_loss = loss_func_gender(gender_pred_probs, y_train_gender)
				race_loss = loss_func_race(race_pred_probs, y_train_race)

				# Apply loss weights
				weighted_loss = age_loss * AGE_LOSS_WEIGHT + gender_loss * GENDER_LOSS_WEIGHT + race_loss * RACE_LOSS_WEIGHT

				optimiser.zero_grad()
				weighted_loss.backward()
				optimiser.step()

			x_val, y_val_age, y_val_gender, y_val_race = next(iter(val_loader))
			model.eval()
			with torch.inference_mode():
				y_val_pred = model(x_val)
			age_val_pred, gender_val_pred_probs, race_val_pred_probs = map(torch.squeeze, y_val_pred)

			age_val_mae = metric_age(age_val_pred, y_val_age).item()
			gender_val_f1 = f1_score(y_val_gender, gender_val_pred_probs.detach().numpy().round())
			race_val_f1 = f1_score(y_val_race.argmax(dim=1), race_val_pred_probs.argmax(dim=1), average='weighted')

			model_path = f'./model_{age_val_mae:.2f}_{gender_val_f1:.2f}_{race_val_f1:.2f}.pth'
			torch.save(model.state_dict(), model_path)

			history['age_val_MAE'].append(age_val_mae)
			history['gender_val_F1'].append(gender_val_f1)
			history['race_val_F1'].append(race_val_f1)

			print(f'Epoch {epoch}/{N_EPOCHS}  |  '
				f'Age val MAE: {age_val_mae:.4f}  |  '
				f'Gender val F1: {gender_val_f1:.4f}  |  '
				f'Race val F1: {race_val_f1:.4f}')

			# Condition early stopping on race val F1 score, as this is the least accurate model output
			if early_stopping(race_val_f1, None):
				print('Early stopping at epoch', epoch)
				break

		# Plot training metrics
		_, (ax_age_val_mae, ax_gender_f1, ax_race_f1) = plt.subplots(nrows=3, sharex=True)
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
		y_test_pred = model(x_test)
	age_test_pred, gender_test_pred_probs, race_test_pred_probs = map(torch.squeeze, y_test_pred)

	print('Test age MAE:', metric_age(age_test_pred, y_test_age).item())

	# ROC curve for gender output

	fpr, tpr, _ = roc_curve(y_test_gender, gender_test_pred_probs)
	plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
	plt.plot(fpr, tpr)
	plt.axis('scaled')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve for gender classification')
	plt.show()

	# Confusion matrices for gender and race outputs

	gender_test_pred_labels = gender_test_pred_probs.round()
	race_test_pred_labels = race_test_pred_probs.argmax(dim=1)
	plot_confusion_matrix('gender', y_test_gender, gender_test_pred_labels, sorted(DATASET_DICT['gender_id'].values()))
	plot_confusion_matrix('race', y_test_race.argmax(dim=1), race_test_pred_labels, sorted(DATASET_DICT['race_id'].values()))

	# Plot first 9 test set images with outputs

	_, axes = plt.subplots(nrows=3, ncols=3)
	plt.subplots_adjust(top=0.8, bottom=0.05, hspace=0.5, wspace=0.4)

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
			f'Predicted: {age_pred}, {gender_pred_label}, {race_pred_label}'
			+ f'\nActual: {int(y_age)}, {y_gender_label}, {y_race_label}'
		)
	plt.suptitle('Test output examples', x=0.508, y=0.95)
	plt.show()
