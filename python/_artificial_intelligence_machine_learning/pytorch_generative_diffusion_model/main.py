"""
PyTorch demo of a generative diffusion model (Denoising Diffusion Probabilistic Model - DDPM)

Author: Sam Barba
Created 28/05/2024
"""

import glob
import os
from time import sleep

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from _utils.plotting import plot_torch_model, plot_image_grid
from diffusion_controller import DiffusionController
from model import DDPM


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

IMG_SIZE = 64
T = 1000  # Diffusion timesteps
BETA_MIN = 1e-4  # Min. noise variance
BETA_MAX = 0.02  # Max. noise variance
T_ENCODING_DIM = 128  # Timestep encoding dim
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_train_loader():
	transform = transforms.Compose([
		transforms.Resize(IMG_SIZE),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalise to [-1,1]
	])

	img_paths = glob.glob('C:/Users/Sam/Desktop/projects/datasets/celeba/*.jpg')
	x = [
		transform(Image.open(img_path)) for img_path in
		tqdm(img_paths, desc='Preprocessing images', unit='imgs', ascii=True)
	]

	dataset = CustomDataset(x)
	train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

	return train_loader


if __name__ == '__main__':
	model = DDPM(num_timesteps=T, encoding_dim=T_ENCODING_DIM, device=DEVICE)
	print(f'\nModel:\n\n{model}')
	plot_torch_model(model, (3, IMG_SIZE, IMG_SIZE), tuple(), input_device=DEVICE)

	diffusion_controller = DiffusionController(num_timesteps=T, beta_min=BETA_MIN, beta_max=BETA_MAX, device=DEVICE)

	if os.path.exists('./model.pth'):
		model.load_state_dict(torch.load('./model.pth', map_location=DEVICE))
	else:
		print('\n----- TRAINING -----\n')

		train_loader = create_train_loader()
		loss_func = torch.nn.MSELoss()
		optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimiser, mode='min', factor=0.5, patience=10, min_lr=1e-5
		)
		early_stopping = EarlyStopping(patience=50, min_delta=0, mode='min')

		# Visualise the forward diffusion process
		first_24_imgs = next(iter(train_loader))[:24].to(DEVICE)
		plot_image_grid(
			first_24_imgs, rows=4, cols=6, gap=4, scale_factor=1.5, scale_interpolation='cubic',
			background_rgb=(0, 0, 0), title_rgb=(255, 255, 255),
			title=f'Forward diffusion process (t=0/{T})',
			save_path='./images/forward_diffusion_step_0.png',
			show=False
		)
		for t in tqdm(range(T), desc='Iterating over forward timesteps', ascii=True):
			t_tensor = torch.full((24,), t, device=DEVICE)
			noisy_images, _ = diffusion_controller.add_noise(first_24_imgs, t_tensor)
			plot_image_grid(
				noisy_images, rows=4, cols=6, gap=4, scale_factor=1.5, scale_interpolation='cubic',
				background_rgb=(0, 0, 0), title_rgb=(255, 255, 255),
				title=f'Forward diffusion process (t={t + 1}/{T})',
				save_path=f'./images/forward_diffusion_step_{(t + 1):0>4}.png',
				show=False
			)

		for epoch in range(1, NUM_EPOCHS + 1):
			progress_bar = tqdm(range(len(train_loader)), unit='batches', ascii=True)
			total_loss = 0

			for img_batch in train_loader:
				progress_bar.update()
				progress_bar.set_description(f'Epoch {epoch}/{NUM_EPOCHS}')

				# Uniformly sample a batch of timesteps
				batch_size = len(img_batch)
				t = torch.randint(0, T, (batch_size,), device=DEVICE)

				# Add noise to images based on sampled timesteps
				noisy_images, noise = diffusion_controller.add_noise(img_batch.to(DEVICE), t)

				# Predict added noise: this is easier than directly predicting the original image from a noisy input.
				# The noise added at each timestep is Gaussian, which is simpler for the model to estimate than the
				# distributions within clean images. Also, training becomes more stable, as the noise distribution is
				# known and relatively uniform, providing consistent gradients and reducing predictive variance.

				noise_pred = model(noisy_images, t)
				loss = loss_func(noise_pred, noise)
				total_loss += loss.item()

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

				progress_bar.set_postfix_str(f'loss={loss.item():.4f}')

			mean_loss = total_loss / len(train_loader)
			progress_bar.set_postfix_str(f"mean_loss={mean_loss:.4f}, lr={optimiser.param_groups[0]['lr']:.3e}")
			progress_bar.close()
			scheduler.step(mean_loss)

			if early_stopping(mean_loss, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

			if DEVICE == 'cuda':
				sleep(20)  # Cooldown

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './model.pth')

	# Test model by denoising a pure noise vector (reverse diffusion process)

	print('\n----- TESTING -----\n')
	model.eval()
	with torch.inference_mode():
		diffusion_controller.generate_images(model, 24, IMG_SIZE)
