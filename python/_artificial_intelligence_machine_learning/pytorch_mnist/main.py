"""
PyTorch MNIST convolutional neural network

Author: Sam Barba
Created 30/10/2022
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist  # Faster to use TF than torchvision
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from _utils.model_architecture_plots import plot_model
from _utils.model_evaluation_plots import plot_confusion_matrix
from conv_net import CNN


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce tensorflow log spam
torch.manual_seed(1)

INPUT_SHAPE = (1, 28, 28)  # Colour channels, H, W
BATCH_SIZE = 256
NUM_EPOCHS = 50
DRAWING_SIZE = 500


def load_data():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Normalise images to [0,1] and correct shape
	x = np.concatenate([x_train, x_test], axis=0).astype(float) / 255
	x = np.reshape(x, (len(x), *INPUT_SHAPE))

	# One-hot encode y
	y = np.concatenate([y_train, y_test])
	y = np.eye(10)[y]  # 10 classes (0-9)

	x, y = torch.tensor(x).float(), torch.tensor(y).float()

	# Create train/validation/test sets (ratio 0.96:0.02:0.02)
	x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, train_size=0.98, stratify=y, random_state=1)
	x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, train_size=0.98, stratify=y_train_val, random_state=1)

	train_set = CustomDataset(x_train, y_train)
	train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)

	return train_loader, x_val, y_val, x_test, y_test


if __name__ == '__main__':
	# 1. Prepare data

	train_loader, x_val, y_val, x_test, y_test = load_data()

	# 2. Define model

	model = CNN()
	print(f'\nModel:\n{model}')
	plot_model(model, INPUT_SHAPE, './images/model_architecture')
	model.to('cpu')

	loss_func = torch.nn.CrossEntropyLoss()

	if os.path.exists('./model.pth'):
		model.load_state_dict(torch.load('./model.pth'))
	else:
		# 3. Plot some example images

		_, axes = plt.subplots(nrows=5, ncols=5, figsize=(5, 5))
		plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.05, wspace=0.05)
		for idx, ax in enumerate(axes.flatten()):
			sample = x_val[idx].squeeze()
			ax.imshow(sample, cmap='gray')
			ax.axis('off')
		plt.suptitle('Data samples', y=0.95)
		plt.show()

		# 4. Train model

		print('\n----- TRAINING -----\n')

		optimiser = torch.optim.Adam(model.parameters())  # LR = 1e-3
		early_stopping = EarlyStopping(patience=5, min_delta=0, mode='max')

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
			with torch.inference_mode():
				y_val_logits = model(x_val)
			val_loss = loss_func(y_val_logits, y_val).item()
			val_f1 = f1_score(y_val.argmax(dim=1), y_val_logits.argmax(dim=1), average='weighted')
			progress_bar.set_postfix_str(f'val_loss={val_loss:.4f}, val_F1={val_f1:.4f}')
			progress_bar.close()

			if early_stopping(val_f1, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './model.pth')

	# 5. Test model

	print('\n----- TESTING -----\n')

	model.eval()
	with torch.inference_mode():
		y_test_logits = model(x_test)
	test_pred = y_test_logits.argmax(dim=1)
	test_loss = loss_func(y_test_logits, y_test)
	print('Test loss:', test_loss.item())

	# Confusion matrix
	f1 = f1_score(y_test.argmax(dim=1), test_pred, average='weighted')
	plot_confusion_matrix(y_test.argmax(dim=1), test_pred, None, f'Test confusion matrix\n(F1 score: {f1:.3f})')

	# User draws a digit to predict

	pg.init()
	pg.display.set_caption('Draw a digit!')
	scene = pg.display.set_mode((DRAWING_SIZE, DRAWING_SIZE))
	user_drawing_coords = []
	drawing = True
	left_btn_down = False

	while drawing:
		for event in pg.event.get():
			match event.type:
				case pg.QUIT:
					drawing = False
					pg.quit()
				case pg.MOUSEBUTTONDOWN:
					if event.button == 1:
						left_btn_down = True
						x, y = event.pos
						user_drawing_coords.append([x, y])
						scene.set_at((x, y), (255, 255, 255))
						pg.display.update()
				case pg.MOUSEMOTION:
					if left_btn_down:
						x, y = event.pos
						user_drawing_coords.append([x, y])
						scene.set_at((x, y), (255, 255, 255))
						pg.display.update()
				case pg.MOUSEBUTTONUP:
					if event.button == 1:
						left_btn_down = False

	user_drawing_coords = np.array(user_drawing_coords) // (DRAWING_SIZE // 27)  # Make coords range from 0-27
	user_drawing_coords = np.unique(user_drawing_coords, axis=0)  # Keep unique pairs only
	drawn_digit_grid = torch.zeros(INPUT_SHAPE[1:])
	drawn_digit_grid[user_drawing_coords[:, 1], user_drawing_coords[:, 0]] = 1
	drawn_digit_input = drawn_digit_grid.reshape((1, *INPUT_SHAPE))
	with torch.inference_mode():
		pred_logits = model(drawn_digit_input)
	pred_probs = torch.softmax(pred_logits, dim=-1)

	plt.imshow(drawn_digit_grid, cmap='gray')
	plt.title(f'Drawn digit is {pred_probs.argmax()} ({(100 * pred_probs.max()):.1f}% sure)')
	plt.axis('off')
	plt.show()
