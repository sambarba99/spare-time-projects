"""
Demo of a variational autoencoder (VAE) for inpainting

Author: Sam Barba
Created 08/09/2024
"""

import glob
import os

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from model import VariationalAutoencoder


torch.manual_seed(1)

IMG_SIZE = 128
CORRUPTED_SQUARE_SIZE = 64
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100


def create_data_loaders():
	def add_black_square(img):
		x1, y1 = torch.randint(0, IMG_SIZE - CORRUPTED_SQUARE_SIZE + 1, size=(2,))
		x2 = x1 + CORRUPTED_SQUARE_SIZE
		y2 = y1 + CORRUPTED_SQUARE_SIZE
		corrupted_img = img.clone()
		corrupted_img[:, y1:y2, x1:x2] = 0

		return corrupted_img, (x1, y1)


	# Preprocess images now instead of during training (faster pipeline overall)

	transform = transforms.Compose([
		transforms.Resize(IMG_SIZE),
		transforms.ToTensor()  # Automatically normalises to [0,1]
	])

	img_paths = glob.glob('C:/Users/Sam/Desktop/projects/datasets/utkface/*.jpg')
	x_ground_truth = [
		transform(Image.open(img_path)) for img_path in
		tqdm(img_paths, desc='Preprocessing images', unit='imgs', ascii=True)
	]
	x_corrputed_and_coords = [
		add_black_square(img) for img in
		tqdm(x_ground_truth, desc='Corrupting images', unit='imgs', ascii=True)
	]
	x_corrputed, corrupted_top_left = zip(*x_corrputed_and_coords)

	# Create train/validation/test sets (ratio 0.96:0.02:0.02)

	indices = torch.arange(len(x_ground_truth))
	train_val_idx, test_idx = train_test_split(indices, train_size=0.98, random_state=1)
	train_idx, val_idx = train_test_split(train_val_idx, train_size=0.98, random_state=1)

	x_train_ground_truth = [x_ground_truth[i] for i in train_idx]
	x_train_corrupted = [x_corrputed[i] for i in train_idx]
	x_val_ground_truth = [x_ground_truth[i] for i in val_idx]
	x_val_corrupted = [x_corrputed[i] for i in val_idx]
	x_test_ground_truth = [x_ground_truth[i] for i in test_idx]
	x_test_corrupted = [x_corrputed[i] for i in test_idx]
	test_corrupted_top_left = [corrupted_top_left[i] for i in test_idx]

	train_dataset = CustomDataset(x_train_corrupted, x_train_ground_truth)
	val_dataset = CustomDataset(x_val_corrupted, x_val_ground_truth)
	test_dataset = CustomDataset(x_test_corrupted, x_test_ground_truth)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
	val_loader = DataLoader(val_dataset, batch_size=len(x_val_ground_truth))
	test_loader = DataLoader(test_dataset, batch_size=len(x_test_ground_truth))

	return train_loader, val_loader, test_loader, test_corrupted_top_left


def plot_images(images, pil_img_transform, title, save_path):
	_, axes = plt.subplots(nrows=4, ncols=6, figsize=(8.4, 6))
	plt.gcf().set_facecolor('black')
	plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, hspace=0.03, wspace=0.03)
	for idx, ax in enumerate(axes.flatten()):
		ax.imshow(pil_img_transform(images[idx]))
		ax.axis('off')
	plt.suptitle(title, y=0.95, color='white')
	plt.savefig(save_path)
	plt.close()


if __name__ == '__main__':
	# Load data

	train_loader, val_loader, test_loader, test_corrupted_top_left = create_data_loaders()

	# Define model

	model = VariationalAutoencoder()
	model.to('cpu')
	print(f'\nModel:\n{model}')

	if os.path.exists('./model.pth'):
		model.load_state_dict(torch.load('./model.pth'))
	else:
		# Train model

		print('\n----- TRAINING -----\n')

		optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
		early_stopping = EarlyStopping(patience=20, min_delta=0, mode='min')

		for epoch in range(1, NUM_EPOCHS + 1):
			progress_bar = tqdm(range(len(train_loader)), unit='batches', ascii=True)
			kl_weight = (epoch - 1) / (NUM_EPOCHS - 1)

			model.train()
			for x_corrputed, x_ground_truth in train_loader:
				progress_bar.update()
				progress_bar.set_description(f'Epoch {epoch}/{NUM_EPOCHS}')

				reconstructed, mu, log_var = model(x_corrputed)
				loss = model.loss(reconstructed, x_ground_truth, mu, log_var, kl_weight)

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

				progress_bar.set_postfix_str(f'loss={loss.item():.2f}')

			model.eval()
			x_val_corrputed, x_val_ground_truth = next(iter(val_loader))
			with torch.inference_mode():
				val_reconstructed, mu, log_var = model(x_val_corrputed)
			val_loss = model.loss(val_reconstructed, x_val_ground_truth, mu, log_var, kl_weight).item()

			progress_bar.set_postfix_str(f'val_loss={val_loss:.2f}')
			progress_bar.close()

			if early_stopping(val_loss, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './model.pth')

	# Plot some test set outputs

	print('\n----- TESTING -----')

	model.eval()
	x_corrputed, x_ground_truth = next(iter(test_loader))
	with torch.inference_mode():
		reconstructed, *_ = model(x_corrputed)

	num_images = 6
	fig, axes = plt.subplots(nrows=3, ncols=num_images, figsize=(10, 4.5))
	plt.subplots_adjust(left=0.17, right=0.98, top=0.95, bottom=0.05, hspace=0, wspace=0.1)
	fig.text(x=0.155, y=0.8, s='Original', ha='right', va='center', fontsize=14)
	fig.text(x=0.155, y=0.5, s='Corrupted', ha='right', va='center', fontsize=14)
	fig.text(x=0.155, y=0.2, s='Reconstructed', ha='right', va='center', fontsize=14)

	pil_image_transform = transforms.ToPILImage()

	for i in range(num_images):
		# Take the reconstructed blacked out square and put it in the ground truth image
		img_reconstructed_temp = reconstructed[i]
		x1, y1 = test_corrupted_top_left[i]
		x2 = x1 + CORRUPTED_SQUARE_SIZE
		y2 = y1 + CORRUPTED_SQUARE_SIZE
		reconstructed_square = img_reconstructed_temp[:, y1:y2, x1:x2]
		img_reconstructed = x_ground_truth[i].clone()
		img_reconstructed[:, y1:y2, x1:x2] = reconstructed_square

		ax1, ax2, ax3 = axes[:, i]
		ax1.imshow(pil_image_transform(x_ground_truth[i]))
		ax2.imshow(pil_image_transform(x_corrputed[i]))
		ax3.imshow(pil_image_transform(img_reconstructed))
		ax1.axis('off')
		ax2.axis('off')
		ax3.axis('off')

	plt.show()

	# Visualise some of the model's latent space by linearly interpolating between 2 random noise vectors

	# z1 = torch.randn(24, 512)
	# z2 = torch.randn_like(z1)
	#
	# for t in torch.linspace(0, 1, 101):
	# 	noise_interp = z1 * t + z2 * (1 - t)
	# 	with torch.inference_mode():
	# 		latent_space_test = model.decoder_block(noise_interp)
	# 	plot_images(
	# 		latent_space_test,
	# 		pil_image_transform,
	# 		f'{t:.2f}(vector_1) + {(1 - t):.2f}(vector_2)',
	# 		f'./images/{t:.2f}.png'
	# 	)
