"""
PyTorch autoencoder demo for the MNIST dataset or for tabular data such as the iris dataset

Author: Sam Barba
Created 17/06/2023
"""

import os

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
import torch

from early_stopping import EarlyStopping
from mnist_autoencoder import MNISTAutoencoder
from my_dataset import MyDataset
from tabular_autoencoder import TabularAutoencoder


pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)
torch.manual_seed(1)

N_EPOCHS = 500

mx = my = 0  # Mouse pointer x and y (for MNIST latent space visualisation)
last_mx = last_my = 0


def do_mnist(batch_size):
	# 1. Prepare data

	(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Faster to use TF than torchvision

	x = np.concatenate([x_train, x_test], axis=0).astype(float)
	y = np.concatenate([y_train, y_test])

	# Normalise images to 0-1 range and correct shape
	x /= 255
	x = np.reshape(x, (len(x), *(1, 28, 28)))  # Colour channels, width, height

	# 2. Load or train model

	model = MNISTAutoencoder().cpu()

	if os.path.exists('./models/mnist_model.pth'):
		model.load_state_dict(torch.load('./models/mnist_model.pth'))
	else:
		train_set = MyDataset(x, y)
		train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
		optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
		loss_func = torch.nn.MSELoss()
		early_stopping = EarlyStopping(patience=5, min_delta=0)
		loss_history = []

		for epoch in range(N_EPOCHS):
			total_loss = 0

			for img_batch, _ in train_loader:  # Don't need labels
				reconstructed = model(img_batch)
				loss = loss_func(reconstructed, img_batch)
				total_loss += loss.item()

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

			loss_history.append(total_loss / batch_size)
			if early_stopping.check_stop(loss_history[-1], model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

			print(f'Epoch {epoch + 1}/{N_EPOCHS}  |  loss = {loss_history[-1]}')

		plt.figure(figsize=(12, 6))
		plt.plot(range(1, N_EPOCHS + 1), loss_history)
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.title('MSE loss per training epoch')
		plt.show()

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './models/mnist_model.pth')

	# 3. Visualise the latent space, controlled by the mouse

	encodings = model.encoder(
		torch.from_numpy(x).float().cpu()
	).detach().numpy()

	plt.rcParams['figure.figsize'] = (12, 6)
	fig, (ax_latent, ax_decoded) = plt.subplots(ncols=2)
	encodings_scatter = ax_latent.scatter(*encodings.T, c=y, s=2, alpha=0.2, cmap=plt.cm.jet)
	handles, _ = encodings_scatter.legend_elements()
	for h in handles:
		h.set_alpha(1)
	ax_latent.legend(handles=handles, labels=range(10))
	ax_latent.set_xlabel('Latent variable 1')
	ax_latent.set_ylabel('Latent variable 2')
	ax_latent.set_title('Latent space')
	ax_decoded.set_title('Decoded image')

	def update_plots(_):
		mouse_scatter = ax_latent.scatter(mx, my, marker='x', color='black', linewidth=2, s=100)

		latent_vector = torch.tensor([mx, my], dtype=torch.float32).unsqueeze(dim=0).cpu()
		decoded_img = model.decoder(latent_vector).detach().numpy().squeeze()
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
	print(f'\nRaw data:\n{df}')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]
	x_to_encode = x.select_dtypes(exclude=np.number).columns
	labels = sorted(y.unique())

	for col in x_to_encode:
		if len(x[col].unique()) > 2:
			one_hot = pd.get_dummies(x[col], prefix=col)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)
		else:  # Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True)

	# Label encode y
	y = y.astype('category').cat.codes.to_frame()
	y.columns = ['classification']

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}')

	x, y = x.to_numpy().astype(float), y.squeeze().to_numpy().astype(int)
	x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))  # Normalise

	return x, y, labels


if __name__ == '__main__':
	choice = input(
		'\nEnter 1 to use MNIST dataset,'
		'\n2 to use banknote dataset,'
		'\n3 for breast tumour dataset,'
		'\n4 for iris dataset,'
		'\n5 for pulsar dataset,'
		'\n6 for Titanic dataset,'
		'\nor 7 for wine dataset\n>>> '
	)

	match choice:
		case '1': path = 'mnist'
		case '2': path = r'C:\Users\Sam\Desktop\Projects\datasets\banknoteData.csv'
		case '3': path = r'C:\Users\Sam\Desktop\Projects\datasets\breastTumourData.csv'
		case '4': path = r'C:\Users\Sam\Desktop\Projects\datasets\irisData.csv'
		case '5': path = r'C:\Users\Sam\Desktop\Projects\datasets\pulsarData.csv'
		case '6': path = r'C:\Users\Sam\Desktop\Projects\datasets\titanicData.csv'
		case _: path = r'C:\Users\Sam\Desktop\Projects\datasets\wineData.csv'

	if path == 'mnist':
		batch_size = 512
		do_mnist(batch_size)
	else:
		# 1. Prepare data

		batch_size = 64
		x, y, labels = load_tabular_data(path)
		n_features_in = x.shape[1]

		choice = input('\nEnter 2 to compress to 2 latent variables, or 3: ')
		n_features_out = int(choice)
		assert n_features_out in (2, 3)  # So can be plotted on xy[z] axes

		# 2. Load or train model

		model_name = path.split('\\')[-1].replace('.csv', '').replace('Data', '')
		model_name = f'{model_name}_model_{n_features_out}_latent_variables'
		model_path = f'./models/{model_name}.pth'

		model = TabularAutoencoder(n_features_in, n_features_out).cpu()

		if os.path.exists(model_path):
			model.load_state_dict(torch.load(model_path))
		else:
			train_set = MyDataset(x, y)
			train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
			optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
			loss_func = torch.nn.MSELoss()
			early_stopping = EarlyStopping(patience=5, min_delta=0)
			loss_history = []

			for epoch in range(N_EPOCHS):
				total_loss = 0

				for batch, _ in train_loader:  # Don't need labels
					reconstructed = model(batch)
					loss = loss_func(reconstructed, batch)
					total_loss += loss.item()

					optimiser.zero_grad()
					loss.backward()
					optimiser.step()

				loss_history.append(total_loss / batch_size)
				if early_stopping.check_stop(loss_history[-1], model.state_dict()):
					print('Early stopping at epoch', epoch)
					break

				print(f'Epoch {epoch + 1}/{N_EPOCHS}  |  loss = {loss_history[-1]}')

			plt.figure(figsize=(12, 6))
			plt.plot(range(1, N_EPOCHS + 1), loss_history)
			plt.xlabel('Epoch')
			plt.ylabel('Loss')
			plt.title('MSE loss per training epoch')
			plt.show()

			model.load_state_dict(early_stopping.best_weights)  # Restore best weights
			torch.save(model.state_dict(), model_path)

		# 3. Visualise the latent space

		encodings = model.encoder(
			torch.from_numpy(x).float().cpu()
		).detach().numpy()

		plt.figure(figsize=(7, 6))
		ax = plt.axes() if n_features_out == 2 else plt.axes(projection='3d')
		scatter = ax.scatter(*encodings.T, c=y, alpha=0.5, cmap=plt.cm.brg) \
			if n_features_out == 2 else \
			ax.scatter3D(*encodings.T, c=y, alpha=0.5, cmap=plt.cm.brg)
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
		# plt.savefig(f'./plots/{model_name}.png')
		plt.show()
