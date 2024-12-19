"""
PyTorch Vision Transformer for MNIST classification

Author: Sam Barba
Created 19/12/2024
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
from torchvision.utils import make_grid
from tqdm import tqdm

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from _utils.plotting import plot_torch_model, plot_image_grid, plot_confusion_matrix
from model import imgs_to_patches, VisionTransformer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce tensorflow log spam
torch.manual_seed(1)

IMG_SIZE = 28
EMBEDDING_DIM = 64
HIDDEN_DIM = EMBEDDING_DIM * 3
NUM_ATTENTION_LAYERS = 3
NUM_HEADS = 8
PATCH_SIZE = 4
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
BATCH_SIZE = 128
NUM_EPOCHS = 50
DRAWING_CELL_SIZE = 15
DRAWING_SIZE = DRAWING_CELL_SIZE * 28
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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

	model = VisionTransformer(
		embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
		num_attention_layers=NUM_ATTENTION_LAYERS, num_heads=NUM_HEADS,
		patch_size=PATCH_SIZE, num_patches=NUM_PATCHES
	).to(DEVICE)
	print(f'\nModel:\n{model}\n')
	plot_torch_model(model, (1, IMG_SIZE, IMG_SIZE), input_device=DEVICE)

	loss_func = torch.nn.CrossEntropyLoss()

	if os.path.exists('./model.pth'):
		model.load_state_dict(torch.load('./model.pth', map_location=DEVICE))
	else:
		# Plot some example images and how they are converted to sequences of patches

		plot_image_grid(
			x_val[:32], rows=4, cols=8, padding=5, scale_factor=2,
			title='Data samples', save_path='./images/data_samples.png'
		)

		img_patches = imgs_to_patches(x_val[:5], PATCH_SIZE, flatten_channels=False)
		img_grids = [
			make_grid(
				patched_img, nrow=IMG_SIZE // PATCH_SIZE, padding=1, normalize=True, pad_value=0.5
			).permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
			for patched_img in img_patches
		]
		plot_image_grid(
			img_grids, rows=1, cols=5, padding=20, scale_factor=3,
			title='Images as input sequences of patches', save_path='./images/imgs_to_patches.png'
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

				x_train = x_train.to(DEVICE)
				y_train = y_train.to(DEVICE)

				y_train_logits = model(x_train)
				loss = loss_func(y_train_logits, y_train)

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

				progress_bar.set_postfix_str(f'loss={loss.item():.4f}')

			model.eval()
			with torch.inference_mode():
				y_val_logits = model(x_val.to(DEVICE)).cpu()
			val_loss = loss_func(y_val_logits, y_val).item()
			val_f1 = f1_score(y_val, y_val_logits.argmax(dim=1), average='weighted')
			progress_bar.set_postfix_str(f'val_loss={val_loss:.4f}, val_F1={val_f1:.4f}')
			progress_bar.close()

			if early_stopping(val_f1, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './model.pth')

	# Test model

	print('\n----- TESTING -----\n')

	model.eval()
	with torch.inference_mode():
		y_test_logits = model(x_test.to(DEVICE)).cpu()
	test_pred = y_test_logits.argmax(dim=1)
	test_loss = loss_func(y_test_logits, y_test)
	print(f'Test loss: {test_loss.item()}')

	# Confusion matrix
	f1 = f1_score(y_test, test_pred, average='weighted')
	plot_confusion_matrix(y_test, test_pred, None, f'Test confusion matrix\n(F1 score: {f1:.3f})')

	# User draws a digit to predict

	pg.init()
	pg.display.set_caption('Draw a digit!')
	scene = pg.display.set_mode((DRAWING_SIZE, DRAWING_SIZE))
	font = pg.font.SysFont('consolas', 16)
	user_drawing_coords = np.zeros((0, 2))
	model_input = torch.zeros((1, IMG_SIZE, IMG_SIZE))
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
			pred_logits = model(model_input.unsqueeze(dim=0).to(DEVICE))
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

	# Given this user input, plot the attention heatmap of the first attention block

	# Convert to a sequence of patches
	img_patches = imgs_to_patches(model_input.unsqueeze(dim=0), patch_size=PATCH_SIZE).to(DEVICE)
	# Pass this through the input layer to get a tensor of size EMBEDDING_DIM
	input_layer_out = model.input_layer(img_patches)
	# Concatenate the classification token and add the positional embedding
	transformer_input = torch.cat([model.cls_token, input_layer_out], dim=1) + model.pos_embedding
	# Pass the embedded image through the first attention block and squeeze the batch dimension (only using 1 image)
	transformer_output = model.transformer[0].fc_block[0](transformer_input).squeeze(0)
	# Reshape the output of the first attention block
	qkv = transformer_output.reshape(NUM_PATCHES + 1, 3, NUM_HEADS, -1)  # Query, key, value
	# Extract the query matrix and permute the dimensions to be (8 heads, 50 patches, 8 channels)
	q = qkv[:, 0].permute(1, 0, 2)
	# Do the same for the key matrix
	k = qkv[:, 1].permute(1, 0, 2)
	kT = k.permute(0, 2, 1)
	# Multiplying q @ kT gives a 8x50x50 matrix showing how much each patch "pays attention" to every other patch
	attention_matrix = q @ kT
	# Average the attention weights across all heads by taking the mean along the first dimension
	attention_matrix_mean = attention_matrix.mean(dim=0)  # 50x50
	# To account for residual connections, we add an identity matrix to the attention matrix
	# and re-normalise the weights (source: https://arxiv.org/abs/2005.00928)
	residual_attention = torch.eye(attention_matrix_mean.shape[1]).to(DEVICE)
	augmented_attention = attention_matrix_mean + residual_attention
	augmented_attention = augmented_attention / augmented_attention.sum(dim=-1).unsqueeze(dim=-1)
	attention_heatmap = augmented_attention[0, 1:].reshape((IMG_SIZE // PATCH_SIZE, IMG_SIZE // PATCH_SIZE))
	attention_heatmap_resized = torch.nn.functional.interpolate(
		attention_heatmap.view(1, 1, *attention_heatmap.shape),
		[IMG_SIZE, IMG_SIZE],
		mode='bilinear'
	)

	_, (ax_digit, ax_heatmap, ax_heatmap_resized) = plt.subplots(ncols=3, figsize=(9, 3))
	ax_digit.imshow(model_input.detach().cpu().numpy().squeeze(), cmap='gray')
	ax_digit.set_title('User-drawn digit')
	ax_digit.axis('off')
	ax_heatmap.imshow(attention_heatmap.detach().cpu().numpy())
	ax_heatmap.set_title('Attention map\n(attention per patch)')
	ax_heatmap.axis('off')
	ax_heatmap_resized.imshow(attention_heatmap_resized.detach().cpu().numpy().squeeze())
	ax_heatmap_resized.set_title('Attention map\n(resized to image size)')
	ax_heatmap_resized.axis('off')
	plt.show()
