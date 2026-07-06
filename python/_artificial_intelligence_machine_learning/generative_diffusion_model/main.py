"""
PyTorch demo of a generative diffusion model (Denoising Diffusion Probabilistic Model - DDPM)

Author: Sam Barba
Created 2024-05-28
"""

from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from _utils.plotting import plot_torch_model
from _utils.progress_bar import ProgressBar
from diffusion_controller import DiffusionController
from model import DDPM


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

IMG_SIZE = 64
T = 1000  # Diffusion timesteps
BETA_MIN = 1e-4  # Min. noise variance
BETA_MAX = 0.02  # Max. noise variance
T_ENCODING_DIM = 128  # Timestep encoding dim
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
NUM_EPOCHS = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_train_loader():
	transform = transforms.Compose([
		transforms.Resize(IMG_SIZE),
		transforms.ToTensor(),  # Scale to [0,1]
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	img_paths = list(Path('C:/Users/sam/Desktop/projects/datasets/celeba').glob('*.jpg'))
	x = [
		transform(Image.open(p)) for p in
		ProgressBar(img_paths, desc='Preprocessing images', unit='imgs')
	]

	dataset = CustomDataset(x)
	train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

	return train_loader


if __name__ == '__main__':
	model = DDPM(num_timesteps=T, encoding_dim=T_ENCODING_DIM, device=DEVICE)
	# Pass in empty tuple for the 't' arg in DDPM forward method
	plot_torch_model(model, (3, IMG_SIZE, IMG_SIZE), tuple(), device=DEVICE)

	diffusion_controller = DiffusionController(num_timesteps=T, beta_min=BETA_MIN, beta_max=BETA_MAX, device=DEVICE)

	if Path('./model.pth').exists():
		model.load_state_dict(torch.load('./model.pth', map_location=DEVICE))
	else:
		train_loader = create_train_loader()

		# Visualise the forward diffusion process
		first_4_imgs = next(iter(train_loader))[:4].to(DEVICE)
		noise_dict = dict.fromkeys(list(range(0, 501, 100)) + [1000])
		noise_dict[0] = first_4_imgs
		for t in list(noise_dict.keys())[1:]:
			t_tensor = torch.full((4,), t - 1, device=DEVICE)
			noisy_images, _ = diffusion_controller.add_noise(first_4_imgs, t_tensor)
			noise_dict[t] = noisy_images

		fig, axes = plt.subplots(nrows=4, ncols=len(noise_dict), figsize=(9, 5))
		plt.subplots_adjust(top=0.84, bottom=0.06, hspace=0, wspace=0.1)
		for idx, (t, noisy_imgs) in enumerate(noise_dict.items()):
			fig.text(x=0.18 + idx / 9, y=0.85, s=f't={t}/1000', ha='center', fontsize=10)
			for i in range(4):
				axes[i, idx].imshow(diffusion_controller.destandardise_transform(noisy_imgs[i]))
				axes[i, idx].axis('off')

		plt.suptitle('Forward diffusion process', y=0.95)
		plt.show()

		print('\n----- TRAINING -----\n')

		loss_func = torch.nn.MSELoss()
		optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=10, min_lr=1e-5)
		early_stopping = EarlyStopping(target=model, patience=50, mode='min')

		for epoch in range(1, NUM_EPOCHS + 1):
			prog_bar = ProgressBar(train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS}', unit='batch', auto_finish=False)
			total_loss = 0

			for img_batch in prog_bar:
				# Uniformly sample a batch of timesteps
				batch_size = img_batch.shape[0]
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

				prog_bar.set_postfix(f'loss={loss.item():.4f}')

			mean_loss = total_loss / len(train_loader)
			prog_bar.finish(f"{mean_loss=:.4f}, lr={optimiser.param_groups[0]['lr']:.3e}")
			scheduler.step(mean_loss)

			if early_stopping(mean_loss):
				early_stopping.print_stop_message()
				break

			if DEVICE == 'cuda':
				sleep(20)  # Cooldown

		early_stopping.restore_best_weights()
		torch.save(model.state_dict(), './model.pth')

	# Test model by denoising a pure noise vector (reverse diffusion process)

	print('\n----- TESTING -----\n')
	model.eval()
	with torch.inference_mode():
		diffusion_controller.generate_images(model, IMG_SIZE)
