"""
PyTorch demo of a Deep Convolutional Generative Adversarial Network (DCGAN)

Author: Sam Barba
Created 01/07/2023
"""

import os

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from _utils.model_architecture_plots import plot_model
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

	root_dir = 'C:/Users/Sam/Desktop/projects/datasets/celeba'
	img_paths = [f'{root_dir}/{f}' for f in os.listdir(root_dir) if f.endswith('.jpg')]
	x = [
		transform(Image.open(fp)) for fp in
		tqdm(img_paths, desc='Preprocessing images', unit='imgs', ascii=True)
	]

	dataset = CustomDataset(x)
	train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

	return train_loader


def plot_images(images, title, save_path=None):
	_, axes = plt.subplots(nrows=4, ncols=6, figsize=(6.16, 4.44))
	plt.gcf().set_facecolor('black')
	plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, hspace=0.03, wspace=0.03)
	for idx, ax in enumerate(axes.flatten()):
		img = (images[idx] + 1) * 127.5  # De-normalise
		img = img.type(torch.uint8).permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
		ax.imshow(img.cpu())
		ax.axis('off')
	plt.suptitle(title, y=0.955, color='white')
	if save_path:
		plt.savefig(save_path)
	else:
		plt.show()
	plt.close()


if __name__ == '__main__':
	gen_model = Generator(latent_dim=GEN_LATENT_DIM)
	disc_model = Discriminator(noise_strength=DISC_NOISE_STRENGTH)

	print(f'\nGenerator model:\n\n{gen_model}')
	print(f'\nDiscriminator model:\n\n{disc_model}')
	plot_model(gen_model, (GEN_LATENT_DIM, 1, 1), './images/generator_architecture')
	plot_model(disc_model, (3, IMG_SIZE, IMG_SIZE), './images/discriminator_architecture')
	gen_model.to(DEVICE)
	disc_model.to(DEVICE)

	if os.path.exists('./gen_model.pth'):
		gen_model.load_state_dict(torch.load('./gen_model.pth', map_location=DEVICE))
	else:
		print('\n----- TRAINING -----\n')

		train_loader = create_train_loader()
		loss_func = torch.nn.BCELoss()
		gen_optimiser = torch.optim.Adam(gen_model.parameters(), lr=LEARNING_RATE, betas=OPTIM_BETAS)
		disc_optimiser = torch.optim.Adam(disc_model.parameters(), lr=LEARNING_RATE, betas=OPTIM_BETAS)
		early_stopping = EarlyStopping(patience=5, min_delta=0, mode='min')

		fixed_noise = torch.randn(24, GEN_LATENT_DIM, 1, 1, device=DEVICE)

		gen_model.eval()
		with torch.inference_mode():
			fake_images_test = gen_model(fixed_noise)
		plot_images(fake_images_test, 'Start', './images/0_start.png')

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
				plot_images(
					fake_images_test,
					f'Epoch {epoch}/{NUM_EPOCHS}, iteration {batch_idx}/{len(train_loader)}',
					f'./images/ep_{epoch:03}_iter_{batch_idx:03}.png'
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
	plot_images(fake_images_test, 'Generator test on random noise')

	# Visualise some of the latent space by linearly interpolating between 2 random noise vectors

	# noise2 = torch.randn_like(noise, device=DEVICE)
	# for t in torch.linspace(0, 1, 101):
	# 	noise_interp = noise * t + noise2 * (1 - t)
	# 	with torch.inference_mode():
	# 		latent_space_test = gen_model(noise_interp)
	# 	plot_images(latent_space_test, f'{t:.2f}(vector_1) + {(1 - t):.2f}(vector_2)', f'./images/{t:.2f}.png')
