"""
Visualising the activations of a PyTorch MNIST classification neural net

Author: Sam Barba
Created 04/11/2024
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
from scipy import ndimage
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist  # Faster to use TF than torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from _utils.model_plotting import plot_cnn_learned_filters


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce tensorflow log spam
torch.manual_seed(1)

# Model
INPUT_SHAPE = (1, 28, 28)  # Colour channels, H, W
BATCH_SIZE = 128
NUM_EPOCHS = 50

# Rendering
DIGIT_CELL_SIZE = 12
DIGIT_CELL_SPACING = 1
DIGIT_CANVAS_SIZE = (DIGIT_CELL_SIZE + DIGIT_CELL_SPACING) * 28
DIGIT_CANVAS_TOP_LEFT_X = 100
DIGIT_CANVAS_TOP_LEFT_Y = 143
CONV_LEFTS = 563, 715
CONV_TOPS = 114, 118
CONV_ZOOMS = 2, 4
CONV_SPACINGS = 1, 9
LINEAR_LEFTS = 873, 1003
LINEAR_TOPS = 107, 172
LINEAR_NODE_RADII = 14, 16
LINEAR_NODE_SPACINGS = 1, 2
CLASS_PROBS_LEFT = 1135
SCENE_WIDTH = 1251
SCENE_HEIGHT = 650


class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
		self.conv2 = nn.Conv2d(8, 8, kernel_size=3)
		self.max_pool = nn.MaxPool2d(2)
		self.dropout = nn.Dropout(0.2)
		self.linear1 = nn.Linear(200, 16)
		self.linear2 = nn.Linear(16, 10)  # 10 classes
		self.leaky_relu = nn.LeakyReLU()

	def forward(self, x):
		conv1_out = self.leaky_relu(self.conv1(x))
		max_pool1_out = self.max_pool(conv1_out)
		conv2_out = self.leaky_relu(self.conv2(max_pool1_out))
		max_pool2_out = self.max_pool(conv2_out)
		flattened = max_pool2_out.flatten(start_dim=1)
		dropout = self.dropout(flattened)
		linear1_out = self.leaky_relu(self.linear1(dropout))
		linear2_out = self.linear2(linear1_out)

		return conv1_out, conv2_out, linear1_out, linear2_out


def load_data():
	(x_train, y_train), (x_val, y_val) = mnist.load_data()

	# Normalise images to [0,1] and add channel dim
	x = np.concatenate([x_train, x_val], axis=0, dtype=float) / 255
	x = np.expand_dims(x, 1)

	# One-hot encode y
	y = np.concatenate([y_train, y_val])
	y = np.eye(10)[y]  # 10 classes (0-9)

	x, y = torch.tensor(x).float(), torch.tensor(y).float()

	# Create train/validation sets (ratio 0.98:0.02)
	x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.98, stratify=y, random_state=1)

	train_set = CustomDataset(x_train, y_train)
	train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)

	return train_loader, x_val, y_val


if __name__ == '__main__':
	# Prepare data

	train_loader, x_val, y_val = load_data()

	# Define model

	model = CNN().cpu()
	print(f'\nModel:\n{model}\n')

	if os.path.exists('./mnist_model.pth'):
		model.load_state_dict(torch.load('./mnist_model.pth'))
	else:
		# Train model

		print('----- TRAINING -----\n')

		loss_func = torch.nn.CrossEntropyLoss()
		optimiser = torch.optim.Adam(model.parameters())  # LR = 1e-3
		early_stopping = EarlyStopping(patience=5, min_delta=0, mode='max')

		for epoch in range(1, NUM_EPOCHS + 1):
			progress_bar = tqdm(range(len(train_loader)), unit='batches', ascii=True)
			model.train()

			for x_train, y_train in train_loader:
				progress_bar.update()
				progress_bar.set_description(f'Epoch {epoch}/{NUM_EPOCHS}')

				*_, y_train_logits = model(x_train)
				loss = loss_func(y_train_logits, y_train)

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

				progress_bar.set_postfix_str(f'loss={loss.item():.4f}')

			model.eval()
			with torch.inference_mode():
				*_, y_val_logits = model(x_val)
			val_loss = loss_func(y_val_logits, y_val).item()
			val_f1 = f1_score(y_val.argmax(dim=1), y_val_logits.argmax(dim=1), average='weighted')
			progress_bar.set_postfix_str(f'val_loss={val_loss:.4f}, val_F1={val_f1:.4f}')
			progress_bar.close()

			if early_stopping(val_f1, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './mnist_model.pth')

	# Plot the model's learned filters
	plot_cnn_learned_filters(model, num_cols=8, figsize=(9, 2))

	# Visualise activations with live drawing

	print('\n----- TESTING -----')

	pg.init()
	pg.display.set_caption('Draw a digit!')
	scene = pg.display.set_mode((SCENE_WIDTH, SCENE_HEIGHT))
	font18 = pg.font.SysFont('consolas', 18)
	font14 = pg.font.SysFont('consolas', 14)
	user_drawing_coords = np.zeros((0, 2))
	model_input = x_val[np.random.choice(len(x_val))].clone()
	left_btn_down = clear_btn_hover = rand_btn_hover = False
	update = True

	while True:
		for event in pg.event.get():
			match event.type:
				case pg.QUIT:
					sys.exit()
				case pg.MOUSEBUTTONDOWN:
					if event.button == 1:
						left_btn_down = True
						x, y = event.pos
						if DIGIT_CANVAS_TOP_LEFT_X <= x <= DIGIT_CANVAS_TOP_LEFT_X + DIGIT_CANVAS_SIZE \
							and DIGIT_CANVAS_TOP_LEFT_Y <= y <= DIGIT_CANVAS_TOP_LEFT_Y + DIGIT_CANVAS_SIZE:
							user_drawing_coords = np.append(user_drawing_coords, [[x, y]], axis=0)
							update = True
						elif 167 <= x <= 236 and 515 <= y <= 545:
							# 'Clear' button clicked
							user_drawing_coords = np.zeros((0, 2))
							model_input = torch.zeros(INPUT_SHAPE)
							update = True
						elif 246 <= x <= 395 and 515 <= y <= 545:
							# 'Random sample' button clicked
							user_drawing_coords = np.zeros((0, 2))
							model_input = x_val[np.random.choice(len(x_val))].clone()
							update = True
				case pg.MOUSEMOTION:
					x, y = event.pos
					if left_btn_down:
						if DIGIT_CANVAS_TOP_LEFT_X <= x <= DIGIT_CANVAS_TOP_LEFT_X + DIGIT_CANVAS_SIZE \
							and DIGIT_CANVAS_TOP_LEFT_Y <= y <= DIGIT_CANVAS_TOP_LEFT_Y + DIGIT_CANVAS_SIZE:
							user_drawing_coords = np.append(user_drawing_coords, [[x, y]], axis=0)
							update = True
					if 167 <= x <= 236 and 515 <= y <= 545:
						clear_btn_hover, rand_btn_hover = True, False
						update = True
					elif 246 <= x <= 395 and 515 <= y <= 545:
						rand_btn_hover, clear_btn_hover = True, False
						update = True
					else:
						update |= (clear_btn_hover | rand_btn_hover)
						clear_btn_hover = rand_btn_hover = False
				case pg.MOUSEBUTTONUP:
					if event.button == 1:
						left_btn_down = False

		if not update:
			continue

		if user_drawing_coords.any():
			# Map coords to range [0,27]
			pixelated_coords = user_drawing_coords - np.array([DIGIT_CANVAS_TOP_LEFT_X, DIGIT_CANVAS_TOP_LEFT_Y])
			pixelated_coords *= 27 / DIGIT_CANVAS_SIZE
			pixelated_coords = np.unique(np.round(pixelated_coords), axis=0).astype(int)  # Keep only unique coords
			pixelated_coords = np.clip(pixelated_coords, 0, 27)

			# Set these pixels as bright
			model_input[:, pixelated_coords[:, 1], pixelated_coords[:, 0]] = 1

			# Add some edge blurring
			for x, y in pixelated_coords:
				for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
					if 0 <= x + dx <= 27 and 0 <= y + dy <= 27 and model_input[:, y + dy, x + dx] == 0:
						model_input[:, y + dy, x + dx] = np.random.uniform(0.33, 1)

		if model_input.any():
			torch.manual_seed(1)
			with torch.inference_mode():
				conv1_out, conv2_out, linear1_out, linear2_out = model(model_input.unsqueeze(dim=0))
			conv1_out = conv1_out.squeeze()
			conv2_out = conv2_out.squeeze()
			linear1_out = linear1_out.squeeze()
			linear2_out = linear2_out.squeeze()
			pred_probs = torch.softmax(linear2_out, dim=-1)
		else:
			conv1_out = torch.zeros(8, 26, 26)
			conv2_out = torch.zeros(8, 11, 11)
			linear1_out = torch.zeros(16)
			linear2_out = pred_probs = torch.zeros(10)

		scene.fill('#0064c8')

		# Render input digit
		for y in range(28):
			for x in range(28):
				colour = round(255 * model_input[0, y, x].item())
				pg.draw.rect(
					scene,
					(colour, colour, colour),
					pg.Rect(
						x * (DIGIT_CELL_SIZE + DIGIT_CELL_SPACING) + DIGIT_CANVAS_TOP_LEFT_X,
						y * (DIGIT_CELL_SIZE + DIGIT_CELL_SPACING) + DIGIT_CANVAS_TOP_LEFT_Y,
						DIGIT_CELL_SIZE,
						DIGIT_CELL_SIZE
					)
				)

		# 'Clear' button
		pg.draw.rect(scene, '#d030e0' if clear_btn_hover else '#003090', pg.Rect(167, 515, 70, 31))
		clear_btn_lbl = font18.render('Clear', True, 'white')
		scene.blit(clear_btn_lbl, (177, 523))

		# 'Random sample' button
		pg.draw.rect(scene, '#d030e0' if rand_btn_hover else '#003090', pg.Rect(246, 515, 150, 31))
		random_btn_lbl = font18.render('Random sample', True, 'white')
		scene.blit(random_btn_lbl, (256, 523))

		# Layer labels
		input_lbl = font14.render('Input', True, 'white')
		conv1_lbl = font14.render('Conv 1', True, 'white')
		conv2_lbl = font14.render('Conv 2', True, 'white')
		linear1_lbl = font14.render('Linear 1', True, 'white')
		linear2_lbl = font14.render('Linear 2', True, 'white')
		softmax_lbl1 = font14.render('Softmax', True, 'white')
		softmax_lbl2 = font14.render('probabilities', True, 'white')
		scene.blit(input_lbl, (261, 116))
		scene.blit(conv1_lbl, (565, 66))
		scene.blit(conv2_lbl, (714, 66))
		scene.blit(linear1_lbl, (841, 66))
		scene.blit(linear2_lbl, (971, 129))
		scene.blit(softmax_lbl1, (1107, 110))
		scene.blit(softmax_lbl2, (1083, 129))

		# Feature maps (activations) of conv layers
		for conv_out, layer_left_pos, layer_top_pos, zoom, spacing \
			in zip([conv1_out, conv2_out], CONV_LEFTS, CONV_TOPS, CONV_ZOOMS, CONV_SPACINGS):
			for idx, feature_map in enumerate(conv_out):
				max_min = feature_map.max() - feature_map.min()
				if max_min:
					feature_map = (feature_map - feature_map.min()) / max_min
				feature_map = (feature_map.numpy() * 255).astype(np.uint8)
				feature_map = ndimage.zoom(feature_map, (zoom, zoom), order=0)
				rgb_arr = np.stack((feature_map, feature_map, feature_map), axis=-1)
				surface = pg.surfarray.make_surface(rgb_arr)
				surface = pg.transform.flip(surface, flip_x=True, flip_y=False)
				surface = pg.transform.rotate(surface, 90)
				scene.blit(surface, (layer_left_pos, layer_top_pos + idx * (len(feature_map) + spacing)))

		# Activations of linear layers
		for layer_out, layer_left_pos, layer_top_pos, node_radius, node_spacing \
			in zip([linear1_out, linear2_out], LINEAR_LEFTS, LINEAR_TOPS, LINEAR_NODE_RADII, LINEAR_NODE_SPACINGS):
			max_min = layer_out.max() - layer_out.min()
			for idx, y in enumerate(layer_out):
				if max_min:
					y_normalised = (y - layer_out.min()) / max_min
					colour = round(255 * y_normalised.item())
				else:
					colour = 0
				pg.draw.circle(
					scene,
					(colour, colour, colour),
					(layer_left_pos, idx * (2 * node_radius + node_spacing) + layer_top_pos),
					node_radius
				)

		# Softmax applied to layer 3
		for idx, p in enumerate(pred_probs):
			colour = round(255 * p.item())
			pg.draw.circle(
				scene,
				(colour, colour, colour),
				(CLASS_PROBS_LEFT, idx * (2 * LINEAR_NODE_RADII[1] + LINEAR_NODE_SPACINGS[1]) + LINEAR_TOPS[1]),
				LINEAR_NODE_RADII[1]
			)
			digit_lbl = font18.render(str(idx), True, '#00c030')
			lbl_rect = digit_lbl.get_rect(
				center=(
					CLASS_PROBS_LEFT,
					idx * (2 * LINEAR_NODE_RADII[1] + LINEAR_NODE_SPACINGS[1]) + LINEAR_TOPS[1]
				)
			)
			scene.blit(digit_lbl, lbl_rect)

		# Layer connections

		# Input to conv 1
		pg.draw.line(
			scene,
			'white',
			(DIGIT_CANVAS_SIZE + DIGIT_CANVAS_TOP_LEFT_X - 26, DIGIT_CANVAS_TOP_LEFT_Y - 5),
			(DIGIT_CANVAS_SIZE + DIGIT_CANVAS_TOP_LEFT_X + 2, DIGIT_CANVAS_TOP_LEFT_Y - 5)
		)
		pg.draw.line(
			scene,
			'white',
			(DIGIT_CANVAS_SIZE + DIGIT_CANVAS_TOP_LEFT_X - 26, DIGIT_CANVAS_SIZE + DIGIT_CANVAS_TOP_LEFT_Y + 3),
			(DIGIT_CANVAS_SIZE + DIGIT_CANVAS_TOP_LEFT_X + 2, DIGIT_CANVAS_SIZE + DIGIT_CANVAS_TOP_LEFT_Y + 3)
		)
		pg.draw.line(
			scene,
			'white',
			(DIGIT_CANVAS_SIZE + DIGIT_CANVAS_TOP_LEFT_X + 3, DIGIT_CANVAS_TOP_LEFT_Y - 5),
			(DIGIT_CANVAS_SIZE + DIGIT_CANVAS_TOP_LEFT_X + 3, DIGIT_CANVAS_SIZE + DIGIT_CANVAS_TOP_LEFT_Y + 3)
		)
		for i in range(8):
			pg.draw.line(
				scene,
				'white',
				(DIGIT_CANVAS_SIZE + DIGIT_CANVAS_TOP_LEFT_X + 4, SCENE_HEIGHT // 2 - 1),
				(
					CONV_LEFTS[0] - 1,
					i * (len(conv1_out[0]) * CONV_ZOOMS[0] + CONV_SPACINGS[0]) + CONV_TOPS[0] + len(conv1_out[0]) - 1
				)
			)

		# Conv 1 to conv 2
		for i in range(8):
			for j in range(8):
				pg.draw.line(
					scene,
					'#e0e0e0',
					(
						CONV_LEFTS[0] + len(conv1_out[0]) * CONV_ZOOMS[0],
						i * (len(conv1_out[0]) * CONV_ZOOMS[0] + CONV_SPACINGS[0]) + CONV_TOPS[0] + len(conv1_out[0]) - 1
					),
					(
						CONV_LEFTS[1] - 1,
						j * (len(conv2_out[0]) * CONV_ZOOMS[1] + CONV_SPACINGS[1]) + CONV_TOPS[1] + len(conv2_out[0]) * CONV_ZOOMS[0] - 1
					)
				)

		# Conv 2 to linear 1
		for i in range(8):
			for j in range(16):
				pg.draw.line(
					scene,
					'#e0e0e0',
					(
						CONV_LEFTS[1] + len(conv2_out[0]) * CONV_ZOOMS[1],
						i * (len(conv2_out[0]) * CONV_ZOOMS[1] + CONV_SPACINGS[1]) + CONV_TOPS[1] + len(conv2_out[0]) * CONV_ZOOMS[0] - 1
					),
					(
						LINEAR_LEFTS[0] - LINEAR_NODE_RADII[0] - 1,
						j * (2 * LINEAR_NODE_RADII[0] + LINEAR_NODE_SPACINGS[0]) + LINEAR_TOPS[0]
					)
				)

		# Linear 1 to linear 2
		for i in range(16):
			for j in range(10):
				pg.draw.line(
					scene,
					'#e0e0e0',
					(
						LINEAR_LEFTS[0] + LINEAR_NODE_RADII[0],
						i * (2 * LINEAR_NODE_RADII[0] + LINEAR_NODE_SPACINGS[0]) + LINEAR_TOPS[0] - 1
					),
					(
						LINEAR_LEFTS[1] - LINEAR_NODE_RADII[1] - 1,
						j * (2 * LINEAR_NODE_RADII[1] + LINEAR_NODE_SPACINGS[1]) + LINEAR_TOPS[1] - 1
					)
				)

		# Linear 2 to softmax output
		for i in range(10):
			pg.draw.line(
				scene,
				'#e0e0e0',
				(
					LINEAR_LEFTS[1] + LINEAR_NODE_RADII[1],
					i * (2 * LINEAR_NODE_RADII[1] + LINEAR_NODE_SPACINGS[1]) + LINEAR_TOPS[1] - 1
				),
				(
					CLASS_PROBS_LEFT - LINEAR_NODE_RADII[1] - 1,
					i * (2 * LINEAR_NODE_RADII[1] + LINEAR_NODE_SPACINGS[1]) + LINEAR_TOPS[1] - 1
				)
			)

		pg.display.update()
		update = False
