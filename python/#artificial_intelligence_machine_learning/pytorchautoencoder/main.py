"""
PyTorch autoencoder for MNIST dataset

Author: Sam Barba
Created 17/06/2023
"""

import os

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
import torch

from autoencoder import Autoencoder
from early_stopping import EarlyStopping
from mnist_dataset import MNISTDataset


plt.rcParams['figure.figsize'] = (12, 6)
torch.manual_seed(1)

BATCH_SIZE = 512
N_EPOCHS = 500


if __name__ == '__main__':
	# 1. Prepare data

	(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Faster to use TF than torchvision

	x = np.concatenate([x_train, x_test], axis=0).astype(float)
	y = np.concatenate([y_train, y_test])

	# Normalise images to 0-1 range and correct shape
	x /= 255
	x = np.reshape(x, (len(x), *(1, 28, 28)))  # Colour channels, width, height

	# 2. Load or train model

	if os.path.exists('model.pth'):
		model = torch.load('model.pth')
	else:
		train_set = MNISTDataset(x, y)
		train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
		model = Autoencoder().cpu()
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

			loss_history.append(total_loss / BATCH_SIZE)
			if early_stopping.check_stop(loss_history[-1], model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

			print(f'Epoch {epoch + 1}/{N_EPOCHS}: loss = {loss_history[-1]:.4f}')

		plt.plot(range(1, N_EPOCHS + 1), loss_history)
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.title('MSE loss per training epoch')
		plt.show()

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model, 'model.pth')

	# 3. Visualise the latent space, controlled by the mouse

	encodings = model.encoder(
		torch.from_numpy(x).float().cpu()
	).detach().numpy()

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

	mx = my = 0  # Mouse pointer x and y
	last_mx = last_my = 0

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
