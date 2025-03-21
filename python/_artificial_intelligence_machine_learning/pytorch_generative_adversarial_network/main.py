"""
PyTorch demo of a Deep Convolutional Generative Adversarial Network (DCGAN)

Author: Sam Barba
Created 01/07/2023
"""

import glob
import os

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from _utils.plotting import plot_torch_model, plot_image_grid
from models import Generator, Discriminator


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

IMG_SIZE = 64
GEN_LATENT_DIM = 128
DISC_NOISE_STRENGTH = 0.05
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
OPTIM_BETAS = (0.5, 0.999)
NUM_EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_train_loader():
	transform = transforms.Compose([
		transforms.Resize(IMG_SIZE),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalise to [-1,1]
	])

	img_paths = glob.glob('C:/Users/sam/Desktop/projects/datasets/celeba/*.jpg')
	x = [
		transform(Image.open(img_path)) for img_path in
		tqdm(img_paths, desc='Preprocessing images', unit='imgs', ascii=True)
	]

	dataset = CustomDataset(x)
	train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

	return train_loader


if __name__ == '__main__':
	gen_model = Generator(latent_dim=GEN_LATENT_DIM).to(DEVICE)
	disc_model = Discriminator(noise_strength=DISC_NOISE_STRENGTH).to(DEVICE)

	print(f'\nGenerator model:\n\n{gen_model}')
	print(f'\nDiscriminator model:\n\n{disc_model}')
	plot_torch_model(
		gen_model, (GEN_LATENT_DIM, 1, 1), input_device=DEVICE, out_file='./images/generator_architecture'
	)
	plot_torch_model(
		disc_model, (3, IMG_SIZE, IMG_SIZE), input_device=DEVICE, out_file='./images/discriminator_architecture'
	)

	if os.path.exists('./gen_model.pth'):
		gen_model.load_state_dict(torch.load('./gen_model.pth', map_location=DEVICE))
	else:
		print('\n----- TRAINING -----\n')

		fixed_noise = torch.randn(24, GEN_LATENT_DIM, 1, 1, device=DEVICE)
		train_loader = create_train_loader()
		loss_func = torch.nn.BCELoss()
		gen_optimiser = torch.optim.Adam(gen_model.parameters(), lr=LEARNING_RATE, betas=OPTIM_BETAS)
		disc_optimiser = torch.optim.Adam(disc_model.parameters(), lr=LEARNING_RATE, betas=OPTIM_BETAS)
		early_stopping = EarlyStopping(patience=5, min_delta=0, mode='min')

		gen_model.eval()
		with torch.inference_mode():
			fake_images_test = gen_model(fixed_noise)
		plot_image_grid(
			fake_images_test, rows=4, cols=6, padding=4, scale_factor=1.5, scale_interpolation='cubic',
			background_rgb=(0, 0, 0), title_rgb=(255, 255, 255),
			title='Start', save_path='./images/0_start.png',
			show=False
		)

		for epoch in range(1, NUM_EPOCHS + 1):
			progress_bar = tqdm(range(len(train_loader)), unit='batches', ascii=True)
			total_gen_loss = 0

			for batch_idx, img_batch in enumerate(train_loader, start=1):
				progress_bar.update()
				progress_bar.set_description(f'Epoch {epoch}/{NUM_EPOCHS}')

				gen_model.train()
				noise = torch.randn(len(img_batch), GEN_LATENT_DIM, 1, 1, device=DEVICE)
				fake = gen_model(noise)

				# Train discriminator

				disc_real = disc_model(img_batch.to(DEVICE))
				disc_real_loss = loss_func(disc_real, torch.ones_like(disc_real))
				disc_fake = disc_model(fake.detach())
				disc_fake_loss = loss_func(disc_fake, torch.zeros_like(disc_fake))
				disc_loss = (disc_real_loss + disc_fake_loss) / 2

				disc_optimiser.zero_grad()
				disc_loss.backward()
				disc_optimiser.step()

				# Train generator

				disc_fake = disc_model(fake)
				gen_loss = loss_func(disc_fake, torch.ones_like(disc_fake))
				total_gen_loss += gen_loss.item()

				gen_optimiser.zero_grad()
				gen_loss.backward()
				gen_optimiser.step()

				progress_bar.set_postfix_str(
					f'disc_real_loss={disc_real_loss.item():.4f}, '
					f'disc_fake_loss={disc_fake_loss.item():.4f}, '
					f'gen_loss={gen_loss.item():.4f}'
				)

				# Plot generation progress

				gen_model.eval()
				with torch.inference_mode():
					fake_images_test = gen_model(fixed_noise)
				plot_image_grid(
					fake_images_test, rows=4, cols=6, padding=4, scale_factor=1.5, scale_interpolation='cubic',
					background_rgb=(0, 0, 0), title_rgb=(255, 255, 255),
					title=f'Epoch {epoch}/{NUM_EPOCHS}, iteration {batch_idx}/{len(train_loader)}',
					save_path=f'./images/ep_{epoch:03}_iter_{batch_idx:03}.png',
					show=False
				)

			mean_gen_loss = total_gen_loss / len(train_loader)
			progress_bar.set_postfix_str(f'mean_gen_loss={mean_gen_loss:.4f}')
			progress_bar.close()

			if early_stopping(mean_gen_loss, gen_model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		gen_model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(gen_model.state_dict(), './gen_model.pth')

	# Test generator on a random noise vector

	noise = torch.randn(24, GEN_LATENT_DIM, 1, 1, device=DEVICE)
	gen_model.eval()
	with torch.inference_mode():
		fake_images_test = gen_model(noise)
	plot_image_grid(
		fake_images_test, rows=4, cols=6, padding=4, scale_factor=1.5, scale_interpolation='cubic',
		background_rgb=(0, 0, 0), title_rgb=(255, 255, 255),
		title='Generator test on random noise'
	)

	# Visualise some of the latent space by linearly interpolating between 2 random noise vectors

	# noise2 = torch.randn_like(noise, device=DEVICE)
	# for t in torch.linspace(0, 1, 101):
	# 	noise_interp = noise * t + noise2 * (1 - t)
	# 	with torch.inference_mode():
	# 		latent_space_test = gen_model(noise_interp)
	# 	plot_image_grid(
	# 		latent_space_test, rows=4, cols=6, padding=4, scale_factor=1.5, scale_interpolation='cubic',
	# 		background_rgb=(0, 0, 0), title_rgb=(255, 255, 255),
	# 		title=f'{t:.2f}(vector_1) + {(1 - t):.2f}(vector_2)',
	# 		save_path=f'./images/{t:.2f}.png',
	# 		show=False
	# 	)
