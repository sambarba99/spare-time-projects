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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.datasets import mnist
import torch

from early_stopping import EarlyStopping
from mnist_autoencoder import MNISTAutoencoder
from tabular_autoencoder import TabularAutoencoder


pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)
torch.manual_seed(1)

N_EPOCHS = 500

# Mouse pointer x and y (for MNIST latent space visualisation)
mx = my = 0
last_mx = last_my = 0


def do_mnist():
	# 1. Prepare data

	(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Faster to use TF than torchvision

	x = np.concatenate([x_train, x_test], axis=0).astype(float)
	y = np.concatenate([y_train, y_test])

	# Normalise images to [0,1] and correct shape
	x = np.reshape(x, (len(x), 1, 28, 28)) / 255  # Colour channels, width, height

	# 2. Load or train model

	model = MNISTAutoencoder()

	if os.path.exists('./models/mnist_model.pth'):
		model.load_state_dict(torch.load('./models/mnist_model.pth'))
	else:
		# Don't need labels (y) as we're autoencoding
		x_train, x_val = train_test_split(x, stratify=y, train_size=0.9, random_state=1)
		x_train = torch.from_numpy(x_train).float()
		x_val = torch.from_numpy(x_val).float()
		batch_size = 500
		optimiser = torch.optim.Adam(model.parameters())  # LR = 1e-3
		loss_func = torch.nn.MSELoss()
		early_stopping = EarlyStopping(patience=10, min_delta=0, mode='min')
		val_loss_history = []

		for epoch in range(1, N_EPOCHS + 1):
			for i in range(0, len(x_train), batch_size):
				x_batch = x_train[i:i + batch_size]

				reconstructed = model(x_batch)
				loss = loss_func(reconstructed, x_batch)

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

			with torch.inference_mode():
				val_reconstructed = model(x_val)
			val_loss = loss_func(val_reconstructed, x_val).item()
			val_loss_history.append(val_loss)

			if epoch % 10 == 0:
				print(f'Epoch {epoch}/{N_EPOCHS}: val MSE = {val_loss}')

			if early_stopping(val_loss, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		plt.figure(figsize=(12, 6))
		plt.plot(range(1, len(val_loss_history) + 1), val_loss_history)
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.title('MSE val loss per training epoch')
		plt.show()

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './models/mnist_model.pth')

	# 3. Visualise the latent space, controlled by the mouse

	encodings = model.encoder_block(
		torch.from_numpy(x).float()
	).detach().numpy()

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

		latent_vector = torch.tensor([mx, my], dtype=torch.float32).unsqueeze(dim=0)
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


def load_tabular_data(path):
	df = pd.read_csv(path)
	print(f'\nRaw data:\n\n{df}')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]
	x_to_encode = x.select_dtypes(exclude=np.number).columns
	labels = sorted(y.unique())

	for col in x_to_encode:
		n_unique = x[col].nunique()
		if n_unique == 1:
			# No information from this feature
			x = x.drop(col, axis=1)
		elif n_unique == 2:
			# Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True).astype(int)
		else:
			# Multivariate feature
			one_hot = pd.get_dummies(x[col], prefix=col).astype(int)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)

	label_encoder = LabelEncoder()
	y = pd.DataFrame(label_encoder.fit_transform(y), columns=['classification'])

	print(f'\nPreprocessed data:\n\n{pd.concat([x, y], axis=1)}')

	x, y = x.to_numpy(), y.to_numpy().squeeze()
	scaler = MinMaxScaler()
	x = scaler.fit_transform(x)

	return x, y, labels


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
		case '2': path = 'C:/Users/Sam/Desktop/Projects/datasets/banknote_authentication.csv'
		case '3': path = 'C:/Users/Sam/Desktop/Projects/datasets/breast_tumour_pathology.csv'
		case '4': path = 'C:/Users/Sam/Desktop/Projects/datasets/glass_classification.csv'
		case '5': path = 'C:/Users/Sam/Desktop/Projects/datasets/iris_classification.csv'
		case '6': path = 'C:/Users/Sam/Desktop/Projects/datasets/mushroom_edibility_classification.csv'
		case '7': path = 'C:/Users/Sam/Desktop/Projects/datasets/pulsar_identification.csv'
		case '8': path = 'C:/Users/Sam/Desktop/Projects/datasets/titanic_survivals.csv'
		case _: path = 'C:/Users/Sam/Desktop/Projects/datasets/wine_classification.csv'

	if path == 'mnist':
		do_mnist()
	else:
		# 1. Prepare data

		x, y, labels = load_tabular_data(path)
		n_features_in = x.shape[1]

		choice = input('\nEnter 2 to compress to 2 latent variables, or 3: ')
		n_features_out = int(choice)
		assert n_features_out in (2, 3)  # So can be plotted on xy[z] axes

		# 2. Load or train model

		model_name = path.split('/')[-1].removesuffix('.csv')
		model_name = f'{model_name}_model_{n_features_out}_latent_variables'
		model_path = f'./models/{model_name}.pth'

		model = TabularAutoencoder(n_features_in, n_features_out)

		if os.path.exists(model_path):
			model.load_state_dict(torch.load(model_path))
		else:
			# Don't need labels (y) as we're autoencoding
			x_train, x_val = train_test_split(x, stratify=y, train_size=0.9, random_state=1)
			x_train = torch.from_numpy(x_train).float()
			x_val = torch.from_numpy(x_val).float()
			batch_size = 64
			optimiser = torch.optim.Adam(model.parameters())  # LR = 1e-3
			loss_func = torch.nn.MSELoss()
			early_stopping = EarlyStopping(patience=50, min_delta=0, mode='min')
			val_loss_history = []

			for epoch in range(1, N_EPOCHS + 1):
				for i in range(0, len(x_train), batch_size):
					x_batch = x_train[i:i + batch_size]

					reconstructed = model(x_batch)
					loss = loss_func(reconstructed, x_batch)

					optimiser.zero_grad()
					loss.backward()
					optimiser.step()

				with torch.inference_mode():
					val_reconstructed = model(x_val)
				val_loss = loss_func(val_reconstructed, x_val).item()
				val_loss_history.append(val_loss)

				if epoch % 10 == 0:
					print(f'Epoch {epoch}/{N_EPOCHS}: val MSE = {val_loss}')

				if early_stopping(val_loss, model.state_dict()):
					print('Early stopping at epoch', epoch)
					break

			plt.figure(figsize=(12, 6))
			plt.plot(range(1, len(val_loss_history) + 1), val_loss_history)
			plt.xlabel('Epoch')
			plt.ylabel('Loss')
			plt.title('MSE val loss per training epoch')
			plt.show()

			model.load_state_dict(early_stopping.best_weights)  # Restore best weights
			torch.save(model.state_dict(), model_path)

		# 3. Visualise the latent space

		encodings = model.encoder_block(
			torch.from_numpy(x).float()
		).detach().numpy()

		plt.figure(figsize=(7, 6))
		ax = plt.axes() if n_features_out == 2 else plt.axes(projection='3d')
		scatter = ax.scatter(*encodings.T, c=y, alpha=0.5, cmap='brg') \
			if n_features_out == 2 else \
			ax.scatter3D(*encodings.T, c=y, alpha=0.5, cmap='brg')
		ax.set_xlabel('Latent variable 1')
		ax.set_ylabel('Latent variable 2')
		if n_features_out == 3:
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
