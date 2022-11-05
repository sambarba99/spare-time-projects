"""
(PyTorch) MNIST convolutional neural network

Author: Sam Barba
Created 30/10/2022
"""

# Reduce TensorFlow logger spam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
import torch
from torch import nn
from torch.utils.data import DataLoader

from conv_net import CNN
from early_stopping import EarlyStopping
from mnist_dataset import MNISTDataset

N_CLASSES = 10  # Class for each digit 0-9
INPUT_SHAPE = (1, 28, 28)  # Colour channels, W, H
DRAWING_SIZE = 500

plt.rcParams['figure.figsize'] = (10, 5)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def load_data(device):
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Normalise images to 0-1 range and correct shape
	x = np.concatenate([x_train, x_test], axis=0).astype(float) / 255
	x = np.reshape(x, (len(x), *INPUT_SHAPE))

	# One-hot encode y
	y = np.concatenate([y_train, y_test])
	y = np.eye(N_CLASSES)[y]

	# Train:validation:test ratio of 0.7:0.2:0.1
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, stratify=y, random_state=1)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.78, stratify=y_train, random_state=1)

	# Convert to tensors
	x_val, y_val, x_test, y_test = map(
		lambda arr: torch.from_numpy(arr).float().to(device),
		[x_val, y_val, x_test, y_test]
	)

	# Val and test sets don't need batching as they're small
	train_set = MNISTDataset(x_train, y_train)
	train_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=8)

	return train_loader, x_val, y_val, x_test, y_test

def plot_confusion_matrix(actual, predictions, labels):
	actual = actual.argmax(dim=1)  # Decode from one-hot

	cm = confusion_matrix(actual, predictions)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
	f1 = f1_score(actual, predictions, average='weighted')

	disp.plot(cmap=plt.cm.plasma)
	plt.title(f'Test confusion matrix\n(F1 score: {f1})')
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	torch.manual_seed(1)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	train_loader, x_val, y_val, x_test, y_test = load_data(device)
	loss_func = nn.CrossEntropyLoss()

	choice = input('\nEnter T to train a new model or L to load existing one\n>>> ').upper()

	match choice:
		case 'T':
			# Plot some training examples

			_, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
			plt.subplots_adjust(hspace=0.01, wspace=0.1)
			for idx, ax in enumerate(axes.flatten()):
				where = np.where(y_val.argmax(axis=1) == idx)[0][0]
				sample = x_val[where].squeeze()
				ax.imshow(sample, cmap='gray')
				ax.set_xticks([])
				ax.set_yticks([])
			plt.suptitle('Data samples', x=0.51, y=0.92)
			plt.show()

			# Build model

			model = CNN(n_classes=N_CLASSES).to(device)
			optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
			print(f'\nModel:\n{model}')

			# Train model

			print('\n----- TRAINING -----\n')

			early_stopping = EarlyStopping(patience=5, min_delta=0)
			history = {'loss': [], 'F1': [], 'val_loss': [], 'val_F1': []}

			for epoch in range(50):
				mean_loss = mean_f1 = 0
				model.train()

				for x, y in train_loader:
					y_probs = model(x).squeeze()
					y_pred = y_probs.argmax(dim=1)

					loss = loss_func(y_probs, y)
					f1 = f1_score(y.argmax(dim=1), y_pred, average='weighted')
					mean_loss += loss.item() / len(train_loader)
					mean_f1 += f1 / len(train_loader)

					optimiser.zero_grad()
					loss.backward()
					optimiser.step()

				model.eval()
				with torch.inference_mode():
					y_val_probs = model(x_val).squeeze()
					y_val_pred = y_val_probs.argmax(dim=1)

					val_loss = loss_func(y_val_probs, y_val)
					val_f1 = f1_score(y_val.argmax(dim=1), y_val_pred, average='weighted')

				history['loss'].append(mean_loss)
				history['F1'].append(mean_f1)
				history['val_loss'].append(val_loss.item())
				history['val_F1'].append(val_f1)

				if epoch % 5 == 0:
					print(f'Epoch: {epoch} | Loss: {mean_loss} | F1: {mean_f1} | Val loss: {val_loss} | Val F1: {val_f1}')

				if early_stopping.check_stop(val_loss, model.state_dict()):
					print('Early stopping at epoch', epoch)
					break

			model.load_state_dict(early_stopping.best_weights)  # Restore best weights

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

			torch.save(model, 'model.pth')
		case 'L':
			model = torch.load('model.pth')
		case _:
			return

	# Testing/evaluation

	print('\n----- TESTING/EVALUATION -----\n')

	model.eval()
	with torch.inference_mode():
		y_test_probs = model(x_test).squeeze()
		test_pred = y_test_probs.argmax(dim=1)

		test_loss = loss_func(y_test_probs, y_test)
		print('Test loss:', test_loss.item())

		plot_confusion_matrix(y_test, test_pred, range(10))

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
	drawn_digit_grid = torch.zeros((28, 28))
	drawn_digit_grid[user_drawing_coords[:, 1], user_drawing_coords[:, 0]] = 1
	drawn_digit_input = drawn_digit_grid.reshape((1, *INPUT_SHAPE)).to(device)
	model.eval()
	with torch.inference_mode():
		pred_vector = model(drawn_digit_input).squeeze()

	plt.imshow(drawn_digit_grid, cmap='gray')
	plt.title(f'Drawn digit is {pred_vector.argmax()} ({(100 * pred_vector.max()):.3f}% sure)')
	plt.show()

if __name__ == '__main__':
	main()
