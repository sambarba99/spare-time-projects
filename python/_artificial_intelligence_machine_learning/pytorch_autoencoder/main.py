"""
PyTorch autoencoder demo for the MNIST dataset or for tabular data e.g. iris dataset

Author: Sam Barba
Created 17/06/2023
"""

import os

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist  # Faster to use TF than torchvision
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from _utils.csv_data_loader import load_csv_classification_data
from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from _utils.model_architecture_plots import plot_model
from mnist_autoencoder import MNISTAutoencoder
from tabular_autoencoder import TabularAutoencoder


pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)
torch.manual_seed(1)

NUM_EPOCHS = 500

# Mouse pointer x and y (for MNIST latent space visualisation)
mx = my = 0
last_mx = last_my = 0


def do_mnist():
	# 1. Prepare data

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x = np.concatenate([x_train, x_test], axis=0).astype(float)
	y = np.concatenate([y_train, y_test])

	# Normalise images to [0,1] and correct shape
	x = np.reshape(x, (len(x), 1, 28, 28)) / 255  # Colour channels, width, height

	x = torch.tensor(x).float()

	# 2. Load or train model

	model = MNISTAutoencoder()
	model.to('cpu')
	plot_model(model, (1, 28, 28), './plots/mnist_autoencoder_architecture')

	if os.path.exists('./models/mnist_model.pth'):
		model.load_state_dict(torch.load('./models/mnist_model.pth'))
	else:
		print('\n----- TRAINING -----\n')

		# Don't need labels (y) as we're autoencoding
		x_train, x_val = train_test_split(x, stratify=y, train_size=0.98, random_state=1)
		train_dataset = CustomDataset(x_train)
		train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
		optimiser = torch.optim.Adam(model.parameters())  # LR = 1e-3
		loss_func = torch.nn.MSELoss()
		early_stopping = EarlyStopping(patience=50, min_delta=0, mode='min')
		val_loss_history = []

		for epoch in range(1, NUM_EPOCHS + 1):
			progress_bar = tqdm(range(len(train_loader)), unit='batches', ascii=True)

			for x_train in train_loader:
				progress_bar.update()
				progress_bar.set_description(f'Epoch {epoch}/{NUM_EPOCHS}')
				reconstructed = model(x_train)
				loss = loss_func(reconstructed, x_train)

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

				progress_bar.set_postfix_str(f'loss={loss.item():.4f}')

			with torch.inference_mode():
				val_reconstructed = model(x_val)
			val_loss = loss_func(val_reconstructed, x_val).item()
			val_loss_history.append(val_loss)
			progress_bar.set_postfix_str(f'{progress_bar.postfix}, val_loss={val_loss:.4f}')
			progress_bar.close()

			if early_stopping(val_loss, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './models/mnist_model.pth')

		plt.figure(figsize=(8, 5))
		plt.plot(range(1, len(val_loss_history) + 1), val_loss_history)
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.title('MSE val loss per training epoch')
		plt.show()

	# 3. Visualise the latent space, controlled by the mouse

	encodings = model.encoder_block(x).detach().numpy()

	fig, (ax_latent, ax_decoded) = plt.subplots(ncols=2, figsize=(9, 5))
	plt.subplots_adjust(wspace=0.1)
	encodings_scatter = ax_latent.scatter(*encodings.T, c=y, s=2, alpha=0.2, cmap='jet')
	handles, _ = encodings_scatter.legend_elements()
	for h in handles:
		h.set_alpha(1)
	ax_latent.legend(handles=handles, labels=range(10))
	ax_latent.axis('scaled')
	ax_latent.set_xlabel('Latent variable 1')
	ax_latent.set_ylabel('Latent variable 2')
	ax_latent.set_title('Latent space')
	ax_decoded.axis('off')
	ax_decoded.set_title('Decoded image')

	def update_plots(_):
		mouse_scatter = ax_latent.scatter(mx, my, marker='x', color='black', linewidth=2, s=100)

		latent_vector = torch.tensor([mx, my]).float().unsqueeze(dim=0)
		decoded_img = model.decoder_block(latent_vector).detach().numpy().squeeze()
		img_plot = ax_decoded.imshow(decoded_img, cmap='gray')

		return mouse_scatter, img_plot

	def on_motion_event(event):
		global mx, my, last_mx, last_my

		try:
			mx = np.clip(event.xdata, encodings[:, 0].min(), encodings[:, 0].max())
			my = np.clip(event.ydata, encodings[:, 1].min(), encodings[:, 1].max())
			last_mx, last_my = mx, my
		except:
			# Mouse out of axis area
			mx, my = last_mx, last_my

	_ = FuncAnimation(fig, update_plots, interval=1, blit=True, cache_frame_data=False)
	fig.canvas.mpl_connect('motion_notify_event', on_motion_event)
	plt.show()


if __name__ == '__main__':
	choice = input(
		'\nEnter 1 to use MNIST dataset,'
		'\n2 to use banknote dataset,'
		'\n3 for breast tumour dataset,'
		'\n4 for glass dataset,'
		'\n5 for iris dataset,'
		'\n6 for mushroom dataset,'
		'\n7 for pulsar dataset,'
		'\n8 for Titanic dataset,'
		'\nor 9 for wine dataset\n>>> '
	)

	match choice:
		case '1': path = 'mnist'
		case '2': path = 'C:/Users/Sam/Desktop/projects/datasets/banknote_authenticity.csv'
		case '3': path = 'C:/Users/Sam/Desktop/projects/datasets/breast_tumour_pathology.csv'
		case '4': path = 'C:/Users/Sam/Desktop/projects/datasets/glass_classification.csv'
		case '5': path = 'C:/Users/Sam/Desktop/projects/datasets/iris_classification.csv'
		case '6': path = 'C:/Users/Sam/Desktop/projects/datasets/mushroom_edibility_classification.csv'
		case '7': path = 'C:/Users/Sam/Desktop/projects/datasets/pulsar_identification.csv'
		case '8': path = 'C:/Users/Sam/Desktop/projects/datasets/titanic_survivals.csv'
		case _: path = 'C:/Users/Sam/Desktop/projects/datasets/wine_classification.csv'

	if path == 'mnist':
		do_mnist()
	else:
		# 1. Prepare data

		x, y, labels, _ = load_csv_classification_data(path, x_transform=MinMaxScaler(), to_tensors=True)
		num_features_in = x.shape[1]

		choice = input('Enter 2 to compress to 2 latent variables, or 3: ')
		num_features_out = int(choice)
		assert num_features_out in (2, 3)  # So can be plotted on xy[z] axes

		# 2. Load or train model

		model_name = path.split('/')[-1].removesuffix('.csv')
		model_name = f'{model_name}_model_{num_features_out}_latent_variables'
		model_path = f'./models/{model_name}.pth'

		model = TabularAutoencoder(num_features_in, num_features_out)
		model.to('cpu')

		if os.path.exists(model_path):
			model.load_state_dict(torch.load(model_path))
		else:
			print('\n----- TRAINING -----\n')

			# Don't need labels (y) as we're autoencoding
			x_train, x_val = train_test_split(x, stratify=y, train_size=0.9, random_state=1)
			train_dataset = CustomDataset(x_train)
			train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
			optimiser = torch.optim.Adam(model.parameters())  # LR = 1e-3
			loss_func = torch.nn.MSELoss()
			early_stopping = EarlyStopping(patience=50, min_delta=0, mode='min')
			val_loss_history = []

			for epoch in range(1, NUM_EPOCHS + 1):
				progress_bar = tqdm(range(len(train_loader)), unit='batches', ascii=True)

				for x_train in train_loader:
					progress_bar.update()
					progress_bar.set_description(f'Epoch {epoch}/{NUM_EPOCHS}')
					reconstructed = model(x_train)
					loss = loss_func(reconstructed, x_train)

					optimiser.zero_grad()
					loss.backward()
					optimiser.step()

					progress_bar.set_postfix_str(f'loss={loss.item():.4f}')

				with torch.inference_mode():
					val_reconstructed = model(x_val)
				val_loss = loss_func(val_reconstructed, x_val).item()
				val_loss_history.append(val_loss)
				progress_bar.set_postfix_str(f'{progress_bar.postfix}, val_loss={val_loss:.4f}')
				progress_bar.close()

				if early_stopping(val_loss, model.state_dict()):
					print('Early stopping at epoch', epoch)
					break

			model.load_state_dict(early_stopping.best_weights)  # Restore best weights
			torch.save(model.state_dict(), model_path)

			plt.figure(figsize=(8, 5))
			plt.plot(range(1, len(val_loss_history) + 1), val_loss_history)
			plt.xlabel('Epoch')
			plt.ylabel('Loss')
			plt.title('MSE val loss per training epoch')
			plt.show()

		# 3. Visualise the latent space

		encodings = model.encoder_block(x).detach().numpy()

		plt.figure(figsize=(7, 6))
		ax = plt.axes() if num_features_out == 2 else plt.axes(projection='3d')
		scatter = ax.scatter(*encodings.T, c=y, alpha=0.5, cmap='brg') \
			if num_features_out == 2 else \
			ax.scatter3D(*encodings.T, c=y, alpha=0.5, cmap='brg')
		ax.set_xlabel('Latent variable 1')
		ax.set_ylabel('Latent variable 2')
		if num_features_out == 3:
			x_plt, y_plt, z_plt = encodings.T
			ax.plot(y_plt, z_plt, 'k.', markersize=2, alpha=0.4, zdir='x', zs=x_plt.min() - 0.1)
			ax.plot(x_plt, z_plt, 'k.', markersize=2, alpha=0.4, zdir='y', zs=y_plt.max() + 0.1)
			ax.plot(x_plt, y_plt, 'k.', markersize=2, alpha=0.4, zdir='z', zs=z_plt.min() - 0.1)
			ax.set_zlabel('Latent variable 3')
		ax.set_title('Latent space')
		handles, _ = scatter.legend_elements()
		for h in handles:
			h.set_alpha(1)
		ax.legend(handles, labels)
		plt.show()
