"""
Visualising the activations of a PyTorch MNIST classification neural net

Author: Sam Barba
Created 04/11/2024
"""

import os
import sys

import numpy as np
import pygame as pg
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist  # Faster to use TF than torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce tensorflow log spam
torch.manual_seed(1)

# Model
INPUT_SHAPE = (1, 28, 28)  # Colour channels, H, W
BATCH_SIZE = 64
NUM_EPOCHS = 50

# Rendering
DIGIT_CELL_SIZE = 12
DIGIT_CELL_SPACING = 1
DIGIT_CANVAS_SIZE = (DIGIT_CELL_SIZE + DIGIT_CELL_SPACING) * 28
DIGIT_CANVAS_TOP_LEFT_X = 100
DIGIT_CANVAS_TOP_LEFT_Y = 143
LAYER_LEFTS = 578, 808, 1038
CLASS_PROBS_LEFT = 1168
LAYER_TOPS = 85, 85, 181
CLASS_PROBS_TOP = 181
LAYER_NODE_RADIUS = 15
LAYER_NODE_SPACING = 2
SCENE_WIDTH = 1283
SCENE_HEIGHT = 650


class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer1 = nn.Linear(784, 16)  # 784 = 1 x 28 x 28
		self.layer2 = nn.Linear(16, 16)
		self.layer3 = nn.Linear(16, 10)  # 10 classes
		self.leaky_relu = nn.LeakyReLU()

	def forward(self, x):
		x_flat = x.flatten(start_dim=1)  # (N, 1, 28, 28) -> (N, 784)
		layer1_out = self.leaky_relu(self.layer1(x_flat)).squeeze()
		layer2_out = self.leaky_relu(self.layer2(layer1_out)).squeeze()
		layer3_out = self.layer3(layer2_out).squeeze()

		return layer1_out, layer2_out, layer3_out


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
	# 1. Prepare data

	train_loader, x_val, y_val = load_data()

	# 2. Define model

	model = Model()
	model.to('cpu')
	print(f'\nModel:\n{model}')

	if os.path.exists('./mnist_model.pth'):
		model.load_state_dict(torch.load('./mnist_model.pth'))
	else:
		# 3. Train model

		print('\n----- TRAINING -----\n')

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

	# 4. Visualise activations with live drawing

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
			with torch.inference_mode():
				layer1_out, layer2_out, layer3_out = model(model_input)
			pred_probs = torch.softmax(layer3_out, dim=-1)
		else:
			layer1_out = layer2_out = torch.zeros(16)
			layer3_out = pred_probs = torch.zeros(10)

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
		layer1_out_lbl = font14.render('Layer 1', True, 'white')
		layer2_out_lbl = font14.render('Layer 2', True, 'white')
		layer3_out_lbl = font14.render('Layer 3', True, 'white')
		softmax_out_lbl1 = font14.render('Softmax', True, 'white')
		softmax_out_lbl2 = font14.render('probabilities', True, 'white')
		scene.blit(input_lbl, (261, 121))
		scene.blit(layer1_out_lbl, (550, 48))
		scene.blit(layer2_out_lbl, (780, 48))
		scene.blit(layer3_out_lbl, (1010, 144))
		scene.blit(softmax_out_lbl1, (1140, 125))
		scene.blit(softmax_out_lbl2, (1116, 144))

		# Activations of layers 1-3
		for layer_out, layer_left_pos, layer_top_pos in zip([layer1_out, layer2_out, layer3_out], LAYER_LEFTS, LAYER_TOPS):
			layer_min, layer_max = layer_out.min(), layer_out.max()
			layer_max_min = layer_max - layer_min
			for idx, y in enumerate(layer_out):
				if layer_max_min != 0:
					y_normalised = (y - layer_min) / layer_max_min
					colour = round(255 * y_normalised.item())
				else:
					colour = 0
				pg.draw.circle(
					scene,
					(colour, colour, colour),
					(layer_left_pos, idx * (2 * LAYER_NODE_RADIUS + LAYER_NODE_SPACING) + layer_top_pos),
					LAYER_NODE_RADIUS
				)

		# Softmax applied to layer 3
		for idx, p in enumerate(pred_probs):
			colour = round(255 * p.item())
			pg.draw.circle(
				scene,
				(colour, colour, colour),
				(CLASS_PROBS_LEFT, idx * (2 * LAYER_NODE_RADIUS + LAYER_NODE_SPACING) + CLASS_PROBS_TOP),
				LAYER_NODE_RADIUS
			)
			digit_lbl = font18.render(str(idx), True, '#00c030')
			lbl_rect = digit_lbl.get_rect(
				center=(
					CLASS_PROBS_LEFT,
					idx * (2 * LAYER_NODE_RADIUS + LAYER_NODE_SPACING) + CLASS_PROBS_TOP
				)
			)
			scene.blit(digit_lbl, lbl_rect)

		# Layer connections

		# Input to layer 1
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
		for i in range(16):
			pg.draw.line(
				scene,
				'#e0e0e0',
				(DIGIT_CANVAS_SIZE + DIGIT_CANVAS_TOP_LEFT_X + 4, SCENE_HEIGHT // 2 - 1),
				(LAYER_LEFTS[0] - LAYER_NODE_RADIUS - 1, i * (2 * LAYER_NODE_RADIUS + LAYER_NODE_SPACING) + LAYER_TOPS[0] - 1)
			)

		# Layer 1 to layer 2
		for i in range(16):
			layer_min, layer_max = layer1_out.min(), layer1_out.max()
			layer_max_min = layer_max - layer_min
			if layer_max_min != 0:
				y_normalised = (layer1_out[i] - layer_min) / layer_max_min
				colour = round(255 * y_normalised.item())
			else:
				colour = 224
			for j in range(16):
				pg.draw.line(
					scene,
					(colour, colour, colour),
					(LAYER_LEFTS[0] + LAYER_NODE_RADIUS, i * (2 * LAYER_NODE_RADIUS + LAYER_NODE_SPACING) + LAYER_TOPS[0] - 1),
					(LAYER_LEFTS[1] - LAYER_NODE_RADIUS - 1, j * (2 * LAYER_NODE_RADIUS + LAYER_NODE_SPACING) + LAYER_TOPS[1] - 1)
				)

		# Layer 2 to layer 3
		for i in range(16):
			layer_min, layer_max = layer2_out.min(), layer2_out.max()
			layer_max_min = layer_max - layer_min
			if layer_max_min != 0:
				y_normalised = (layer2_out[i] - layer_min) / layer_max_min
				colour = round(255 * y_normalised.item())
			else:
				colour = 224
			for j in range(10):
				pg.draw.line(
					scene,
					(colour, colour, colour),
					(LAYER_LEFTS[1] + LAYER_NODE_RADIUS, i * (2 * LAYER_NODE_RADIUS + LAYER_NODE_SPACING) + LAYER_TOPS[1] - 1),
					(LAYER_LEFTS[2] - LAYER_NODE_RADIUS - 1, j * (2 * LAYER_NODE_RADIUS + LAYER_NODE_SPACING) + LAYER_TOPS[2] - 1)
				)

		# Layer 3 to softmax output
		for i in range(10):
			pg.draw.line(
				scene,
				'#e0e0e0',
				(LAYER_LEFTS[2] + LAYER_NODE_RADIUS, i * (2 * LAYER_NODE_RADIUS + LAYER_NODE_SPACING) + LAYER_TOPS[2] - 1),
				(CLASS_PROBS_LEFT - LAYER_NODE_RADIUS - 1, i * (2 * LAYER_NODE_RADIUS + LAYER_NODE_SPACING) + CLASS_PROBS_TOP - 1)
			)

		pg.display.update()
		update = False
