"""
PyTorch MNIST convolutional neural network

Author: Sam Barba
Created 30/10/2022
"""

import os

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
from _utils.plotting import *
from conv_net import CNN


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce tensorflow log spam
torch.manual_seed(1)

INPUT_SHAPE = (1, 28, 28)  # Colour channels, H, W
BATCH_SIZE = 256
NUM_EPOCHS = 50
DRAWING_CELL_SIZE = 15
DRAWING_SIZE = DRAWING_CELL_SIZE * 28


def load_data():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Normalise images to [0,1] and add channel dim
	x = np.concatenate([x_train, x_test], axis=0, dtype=float) / 255
	x = np.expand_dims(x, 1)

	y = np.concatenate([y_train, y_test])

	x, y = torch.tensor(x).float(), torch.tensor(y).long()

	# Create train/validation/test sets (ratio 0.96:0.02:0.02)
	x_train_val, x_test, y_train_val, y_test = train_test_split(
		x, y, train_size=0.98, stratify=y, random_state=1
	)
	x_train, x_val, y_train, y_val = train_test_split(
		x_train_val, y_train_val, train_size=0.98, stratify=y_train_val, random_state=1
	)

	train_set = CustomDataset(x_train, y_train)
	train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)

	return train_loader, x_val, y_val, x_test, y_test


if __name__ == '__main__':
	# Prepare data

	train_loader, x_val, y_val, x_test, y_test = load_data()

	# Define model

	model = CNN().cpu()
	print(f'\nModel:\n{model}\n')
	plot_torch_model(model, INPUT_SHAPE)

	loss_func = torch.nn.CrossEntropyLoss()

	if os.path.exists('./model.pth'):
		model.load_state_dict(torch.load('./model.pth'))
	else:
		# Plot some example images

		plot_image_grid(
			x_val[:32], rows=4, cols=8, padding=5, scale_factor=2,
			title='Data samples', save_path='./images/data_samples.png'
		)

		# Train model

		print('----- TRAINING -----\n')

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
			val_f1 = f1_score(y_val, y_val_logits.argmax(dim=1), average='weighted')
			progress_bar.set_postfix_str(f'val_loss={val_loss:.4f}, val_F1={val_f1:.4f}')
			progress_bar.close()

			if early_stopping(val_f1, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './model.pth')

	# Plot the model's learned filters
	layer_filters = get_cnn_learned_filters(model)
	for idx, (filters, padding) in enumerate(zip(layer_filters, (15, 10)), start=1):
		cols = 8
		rows = len(filters) // cols
		plot_image_grid(
			filters, rows, cols, padding=padding, scale_factor=20,
			title=f'Filters of conv layer {idx}/{len(layer_filters)}',
			save_path=f'./images/conv{idx}_filters.png'
		)

	# Test model

	print('\n----- TESTING -----\n')

	model.eval()
	with torch.inference_mode():
		y_test_logits = model(x_test)
	test_pred = y_test_logits.argmax(dim=1)
	test_loss = loss_func(y_test_logits, y_test)
	print(f'Test loss: {test_loss.item()}\n')

	# Confusion matrix
	f1 = f1_score(y_test, test_pred, average='weighted')
	plot_confusion_matrix(y_test, test_pred, None, f'Test confusion matrix\n(F1 score: {f1:.3f})')

	# User draws a digit to predict

	pg.init()
	pg.display.set_caption('Draw a digit!')
	scene = pg.display.set_mode((DRAWING_SIZE, DRAWING_SIZE))
	font = pg.font.SysFont('consolas', 16)
	user_drawing_coords = np.zeros((0, 2))
	model_input = torch.zeros(INPUT_SHAPE)
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
						user_drawing_coords = np.append(user_drawing_coords, [[x, y]], axis=0)
				case pg.MOUSEMOTION:
					if left_btn_down:
						x, y = event.pos
						user_drawing_coords = np.append(user_drawing_coords, [[x, y]], axis=0)
				case pg.MOUSEBUTTONUP:
					if event.button == 1:
						left_btn_down = False

		if not left_btn_down:
			continue

		# Map coords to range [0,27]
		pixelated_coords = user_drawing_coords * 27 / DRAWING_SIZE
		pixelated_coords = np.unique(pixelated_coords.round(), axis=0).astype(int)  # Keep only unique coords
		pixelated_coords = np.clip(pixelated_coords, 0, 27)

		# Set these pixels as bright
		model_input[:, pixelated_coords[:, 1], pixelated_coords[:, 0]] = 1

		# Add some edge blurring
		for x, y in pixelated_coords:
			for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
				if 0 <= x + dx <= 27 and 0 <= y + dy <= 27 and model_input[:, y + dy, x + dx] == 0:
					model_input[:, y + dy, x + dx] = np.random.uniform(0.33, 1)

		with torch.inference_mode():
			pred_logits = model(model_input.unsqueeze(dim=0))
		pred_probs = torch.softmax(pred_logits, dim=-1)

		for y in range(28):
			for x in range(28):
				colour = round(255 * model_input[0, y, x].item())
				pg.draw.rect(
					scene,
					(colour, colour, colour),
					pg.Rect(x * DRAWING_CELL_SIZE, y * DRAWING_CELL_SIZE, DRAWING_CELL_SIZE, DRAWING_CELL_SIZE)
				)

		pred_lbl = font.render(f'{pred_probs.argmax()} ({(100 * pred_probs.max()):.1f}% sure)', True, 'green')
		scene.blit(pred_lbl, (10, 10))

		pg.display.update()

	# Plot feature maps for user-drawn digit
	layer_feature_maps = get_cnn_feature_maps(model, input_img=model_input)
	for idx, (feature_map, padding, scale_factor) in enumerate(zip(layer_feature_maps, (15, 10), (3, 6)), start=1):
		cols = 8
		rows = len(feature_map) // cols
		plot_image_grid(
			feature_map, rows, cols, padding=padding, scale_factor=scale_factor,
			title=f'Feature map of conv layer {idx}/{len(layer_feature_maps)} (user-drawn digit)',
			save_path=f'./images/conv{idx}_feature_map.png'
		)
