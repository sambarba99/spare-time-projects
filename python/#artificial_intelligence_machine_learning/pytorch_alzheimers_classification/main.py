"""
PyTorch classification of brain scans for Alzheimer's disease detection

Author: Sam Barba
Created 27/01/2024
"""

import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from conv_net import CNN
from custom_dataset import CustomDataset
from early_stopping import EarlyStopping


plt.rcParams['figure.figsize'] = (9, 6)
np.random.seed(1)
pd.set_option('display.width', None)
pd.set_option('max_colwidth', None)
torch.manual_seed(1)

DATA_PATH = 'C:/Users/Sam/Desktop/Projects/datasets/alzheimers'  # Available from Kaggle
DATA_SUBFOLDERS = ['0_healthy', '1_very_mild', '2_mild', '3_moderate']  # Also class names
IMG_H = 208
IMG_W = 176
SCALE_FACTOR = 0.5
INPUT_H = round(IMG_H * SCALE_FACTOR)
INPUT_W = round(IMG_W * SCALE_FACTOR)
BATCH_SIZE = 128
N_EPOCHS = 100


def load_data(df):
	def preprocess_img(path):
		img = cv.imread(path)
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		img = cv.resize(img, (INPUT_W, INPUT_H))
		img = img.astype(np.float64) / 255  # Normalise to [0,1]
		img = img.reshape(1, INPUT_H, INPUT_W)  # Colour channels, H, W

		return img


	x = np.array([
		preprocess_img(p) for p in
		tqdm(df['img_path'], desc='Preprocessing images', ascii=True)
	])
	y = pd.get_dummies(df['class'], prefix='class').astype(int).to_numpy()

	# Train:validation:test ratio of 0.8:0.1:0.1
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.89, stratify=y, random_state=1)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.89, stratify=y_train, random_state=1)

	# Convert to tensors
	x_val, y_val, x_test, y_test = map(
		lambda arr: torch.from_numpy(arr).float(),
		[x_val, y_val, x_test, y_test]
	)

	train_set = CustomDataset(x_train, y_train)
	train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

	return train_loader, x_val, y_val, x_test, y_test


if __name__ == '__main__':
	# 1. Convert data to dataframe

	data = []
	for subfolder in DATA_SUBFOLDERS:
		directory = f'{DATA_PATH}/{subfolder}'
		for img_path in os.listdir(directory):
			data.append((f'{directory}/{img_path}', subfolder))  # subfolder = class name

	df = pd.DataFrame(data, columns=['img_path', 'class'])

	# 2. Plot some examples

	example_indices = [0, 1, 3200, 3201, 5440, 5441, 6336, 6337]
	_, axes = plt.subplots(nrows=2, ncols=4)
	plt.subplots_adjust(top=0.85, bottom=0.05, hspace=0, wspace=0.05)
	for idx, ax in zip(example_indices, axes.flatten()):
		sample = cv.imread(df['img_path'][idx])
		ax.imshow(sample)
		ax.axis('off')
		ax.set_title(df['class'][idx][2:].replace('_', ' '))
	plt.suptitle('Data samples', x=0.514, y=0.94)
	plt.show()

	# 3. Plot output feature (class) distributions

	unique_values_counts = df['class'].value_counts()
	plt.bar(unique_values_counts.index, unique_values_counts.values)
	plt.xlabel('Class')
	plt.ylabel('Count')
	plt.title('Class distribution')
	plt.show()

	# Fix some of the class imbalance by adding duplicates

	class_2_rows = df[df['class'] == '2_mild']
	class_3_rows = df[df['class'] == '3_moderate']
	df = pd.concat([df] + [class_2_rows], ignore_index=True)
	df = pd.concat([df] + [class_3_rows] * 27, ignore_index=True)

	# 4. Preprocess data for loading

	print(f'\nRaw data:\n{df}\n')

	train_loader, x_val, y_val, x_test, y_test = load_data(df)
	loss_func = torch.nn.CrossEntropyLoss()
	model = CNN()
	print(f'\nModel:\n{model}')

	if os.path.exists('./model.pth'):
		model.load_state_dict(torch.load('./model.pth'))
	else:
		# 5. Train model

		print('\n----- TRAINING -----\n')

		optimiser = torch.optim.Adam(model.parameters(), lr=2e-4)
		early_stopping = EarlyStopping(patience=10, min_delta=0, mode='max')
		history = {'loss': [], 'F1': [], 'val_loss': [], 'val_F1': []}

		for epoch in range(1, N_EPOCHS + 1):
			total_loss = total_f1 = 0
			model.train()

			for x, y in train_loader:
				y_probs = model(x).squeeze()
				y_pred = y_probs.argmax(dim=1)

				loss = loss_func(y_probs, y)
				f1 = f1_score(y.argmax(dim=1), y_pred, average='weighted')
				total_loss += loss.item()
				total_f1 += f1

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

			model.eval()
			with torch.inference_mode():
				y_val_probs = model(x_val).squeeze()
				y_val_pred = y_val_probs.argmax(dim=1)

				val_loss = loss_func(y_val_probs, y_val).item()
				val_f1 = f1_score(y_val.argmax(dim=1), y_val_pred, average='weighted')

			history['loss'].append(total_loss / len(train_loader))
			history['F1'].append(total_f1 / len(train_loader))
			history['val_loss'].append(val_loss)
			history['val_F1'].append(val_f1)

			print(f'Epoch {epoch}/{N_EPOCHS} | '
				f'Loss: {total_loss / len(train_loader)} | '
				f'F1: {total_f1 / len(train_loader)} | '
				f'Val loss: {val_loss} | '
				f'Val F1: {val_f1}')

			if early_stopping(val_f1, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './model.pth')

		# Plot loss and F1 throughout training
		_, (ax_loss, ax_f1) = plt.subplots(nrows=2, sharex=True)
		ax_loss.plot(history['loss'], label='Training loss')
		ax_loss.plot(history['val_loss'], label='Validation loss')
		ax_f1.plot(history['F1'], label='Training F1')
		ax_f1.plot(history['val_F1'], label='Validation F1')
		ax_loss.set_ylabel('Categorical\ncross-entropy')
		ax_f1.set_ylabel('F1')
		ax_f1.set_xlabel('Epoch')
		ax_loss.legend()
		ax_f1.legend()
		plt.title('Model loss and F1 score during training', y=2.24)
		plt.show()

	# 6. Testing

	print('\n----- TESTING -----\n')

	model.eval()
	with torch.inference_mode():
		y_test_probs = model(x_test).squeeze()
		test_pred = y_test_probs.argmax(dim=1)

		test_loss = loss_func(y_test_probs, y_test)
		print('Test loss:', test_loss.item())

		# Confusion matrix

		cm = confusion_matrix(y_test.argmax(dim=1), test_pred)
		disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DATA_SUBFOLDERS)
		f1 = f1_score(y_test.argmax(dim=1), test_pred, average='weighted')

		disp.plot(cmap='plasma')
		plt.title(f'Test confusion matrix\n(F1 score: {f1})')
		plt.show()
