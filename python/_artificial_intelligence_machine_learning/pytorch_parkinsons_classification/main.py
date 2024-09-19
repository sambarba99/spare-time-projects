"""
PyTorch classification of patient drawings for Parkinson's disease detection

Author: Sam Barba
Created 27/01/2024
"""

import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
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

DATA_PATH = 'C:/Users/Sam/Desktop/projects/datasets/parkinsons'
DATA_SUBFOLDERS = ['spiral_healthy', 'spiral_parkinsons', 'wave_healthy', 'wave_parkinsons']
IMG_SIZE = 64
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100


def create_data_loaders(df):
	def preprocess_img(path, target_w_to_h=1):
		img = cv.imread(path)
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		img = cv.medianBlur(img, 3)  # Denoise
		_, img = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)

		# Make img aspect ratio 1:1
		img_h, img_w = img.shape[:2]

		if img_w / img_h < target_w_to_h:
			# Extend width of image with black background
			new_w = img_h
			new_img = np.zeros((img_h, new_w), dtype=np.uint8)
			horiz_paste_pos = (new_w - img_w) // 2
			new_img[:, horiz_paste_pos:horiz_paste_pos + img_w] = img
		elif img_w / img_h > target_w_to_h:
			# Extend height of image with black background
			new_h = img_w
			new_img = np.zeros((new_h, img_w), dtype=np.uint8)
			vert_paste_pos = (new_h - img_h) // 2
			new_img[vert_paste_pos:vert_paste_pos + img_h] = img
		else:
			new_img = img.copy()

		# Dilate drawing so that resizing doesn't "damage" it later
		kernel = np.ones((4, 4), np.uint8)
		new_img = cv.dilate(new_img, kernel)

		new_img = cv.resize(new_img, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_NEAREST)

		return new_img


	x = [
		preprocess_img(p) for p in
		tqdm(df['img_path'], desc='Preprocessing images', unit='imgs', ascii=True)
	]
	y = pd.get_dummies(df['class'], prefix='class', drop_first=True, dtype=int).to_numpy().squeeze()
	class_dict = {'healthy': 0, 'parkinsons': 1}

	x_augmented = []
	for idx, img in enumerate(x):
		x_augmented.append(cv.rotate(img, cv.ROTATE_90_CLOCKWISE))
		x_augmented.append(cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE))
		x_augmented.append(cv.rotate(img, cv.ROTATE_180))
		x_augmented.append(cv.flip(img, 0))
		x_augmented.append(cv.flip(img, 1))
		y = np.append(y, [class_dict[df['class'][idx]]] * 5)

	x.extend(x_augmented)
	x = [img.astype(np.float64) / 255 for img in x]  # Normalise to [0,1]
	x = [img.reshape(1, IMG_SIZE, IMG_SIZE) for img in x]  # Colour channels, H, W
	x = [torch.tensor(xi).float() for xi in x]
	y = torch.tensor(y).float()

	# Create train/validation/test sets (ratio 0.7:0.2:0.1)
	x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, train_size=0.9, stratify=y, random_state=1)
	x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, train_size=0.78, stratify=y_train_val, random_state=1)

	train_set = CustomDataset(x_train, y_train)
	val_set = CustomDataset(x_val, y_val)
	test_set = CustomDataset(x_test, y_test)
	train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
	val_loader = DataLoader(val_set, batch_size=len(x_val))
	test_loader = DataLoader(test_set, batch_size=len(x_test))

	return train_loader, val_loader, test_loader


if __name__ == '__main__':
	# 1. Convert data to dataframe

	data = []
	for subfolder in DATA_SUBFOLDERS:
		directory = f'{DATA_PATH}/{subfolder}'
		class_name = subfolder.split('_')[1]
		for img_path in os.listdir(directory):
			data.append((f'{directory}/{img_path}', class_name))

	df = pd.DataFrame(data, columns=['img_path', 'class'])

	# 2. Plot some examples

	example_indices = [0, 52, 102, 154]
	_, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 5))
	plt.subplots_adjust(top=0.85, bottom=0, hspace=0.1, wspace=0.1)
	for idx, ax in zip(example_indices, axes.flatten()):
		sample = cv.imread(df['img_path'][idx])
		ax.imshow(sample)
		ax.axis('off')
		ax.set_title(df['class'][idx])
	plt.suptitle('Data samples', x=0.514, y=0.95)
	plt.show()

	# 3. Plot output feature (class) distributions

	unique_values_counts = df['class'].value_counts()
	plt.bar(unique_values_counts.index, unique_values_counts.values)
	plt.xlabel('Class')
	plt.ylabel('Count')
	plt.title('Class distribution')
	plt.show()

	# 4. Preprocess data for loading

	print(f'\nRaw data:\n{df}\n')

	train_loader, val_loader, test_loader = create_data_loaders(df)

	model = CNN()
	print(f'\nModel:\n{model}')
	plot_model(model, (1, IMG_SIZE, IMG_SIZE), './images/model_architecture')
	model.to('cpu')

	loss_func = torch.nn.BCEWithLogitsLoss()

	if os.path.exists('./model.pth'):
		model.load_state_dict(torch.load('./model.pth'))
	else:
		# 5. Train model

		print('\n----- TRAINING -----\n')

		optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
		early_stopping = EarlyStopping(patience=10, min_delta=0, mode='max')

		for epoch in range(1, NUM_EPOCHS + 1):
			progress_bar = tqdm(range(len(train_loader)), unit='batches', ascii=True)
			model.train()

			for x_train, y_train in train_loader:
				progress_bar.update()
				progress_bar.set_description(f'Epoch {epoch}/{NUM_EPOCHS}')

				y_train_logits = model(x_train)
				loss = loss_func(y_train_logits, y_train)

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

				progress_bar.set_postfix_str(f'loss={loss.item():.4f}')

			model.eval()
			x_val, y_val = next(iter(val_loader))
			with torch.inference_mode():
				y_val_logits = model(x_val)
			y_val_probs = torch.sigmoid(y_val_logits)
			y_val_pred = y_val_probs.round()
			val_loss = loss_func(y_val_logits, y_val).item()
			val_f1 = f1_score(y_val, y_val_pred)
			progress_bar.set_postfix_str(f'val_loss={val_loss:.4f}, val_F1={val_f1:.4f}')
			progress_bar.close()

			if early_stopping(val_f1, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './model.pth')

	# 6. Test model (plot confusion matrix and ROC curve)

	print('\n----- TESTING -----\n')

	model.eval()
	x_test, y_test = next(iter(test_loader))
	with torch.inference_mode():
		y_test_logits = model(x_test)
	y_test_probs = torch.sigmoid(y_test_logits)
	y_test_pred = y_test_probs.round()
	test_loss = loss_func(y_test_logits, y_test)
	print('Test loss:', test_loss.item())

	f1 = f1_score(y_test, y_test_pred)
	plot_confusion_matrix(y_test, y_test_pred, ['healthy', 'parkinsons'], f'Test confusion matrix\n(F1 score: {f1:.3f})')

	plot_roc_curve(y_test, y_test_logits)
