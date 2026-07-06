"""
PyTorch demo of a Deep Convolutional Generative Adversarial Network (DCGAN)

Author: Sam Barba
Created 2023-07-01
"""

from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from _utils.custom_dataset import CustomDataset
from _utils.plotting import plot_image_grid, plot_torch_model
from _utils.progress_bar import ProgressBar
from models import Generator, Discriminator


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

IMG_SIZE = 64
GEN_LATENT_DIM = 128
DISC_NOISE_STRENGTH = 0.05
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
OPTIM_BETAS = (0.5, 0.999)
NUM_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

destandardise_transform = transforms.Lambda(lambda img: img * 0.5 + 0.5)


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


def plot_output(save_path):
	with torch.inference_mode():
		fake = gen_model(fixed_noise)
	plot_image_grid(
		fake, rows=4, cols=6, padding=4, transform=destandardise_transform, scale_factor=1.5,
		scale_interpolation='cubic', background_rgb=(0, 0, 0), title_rgb=(255, 255, 255),
		save_path=save_path,
		show='epoch' not in save_path
	)


if __name__ == '__main__':
	gen_model = Generator(latent_dim=GEN_LATENT_DIM).to(DEVICE)
	disc_model = Discriminator(noise_strength=DISC_NOISE_STRENGTH).to(DEVICE)

	# For generation
	num_images = 24
	fixed_noise = torch.randn(num_images, GEN_LATENT_DIM, 1, 1, device=DEVICE)

	plot_torch_model(
		gen_model, (GEN_LATENT_DIM, 1, 1), device=DEVICE, out_file='./images/generator_architecture'
	)
	plot_torch_model(
		disc_model, (3, IMG_SIZE, IMG_SIZE), device=DEVICE, out_file='./images/discriminator_architecture'
	)

	if Path('./gen_model.pth').exists():
		gen_model.load_state_dict(torch.load('./gen_model.pth', map_location=DEVICE))
	else:
		train_loader = create_train_loader()
		loss_func = torch.nn.BCEWithLogitsLoss()
		gen_optimiser = torch.optim.Adam(gen_model.parameters(), lr=LEARNING_RATE, betas=OPTIM_BETAS)
		disc_optimiser = torch.optim.Adam(disc_model.parameters(), lr=LEARNING_RATE, betas=OPTIM_BETAS)
		disc_model.train()

		print('\n----- TRAINING -----\n')

		for epoch in range(1, NUM_EPOCHS + 1):
			gen_model.train()

			prog_bar = ProgressBar(train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS}', unit='batch', auto_finish=False)
			total_disc_loss = total_gen_loss = 0

			for img_batch in prog_bar:
				# Train discriminator

				noise = torch.randn(img_batch.shape[0], GEN_LATENT_DIM, 1, 1, device=DEVICE)
				with torch.no_grad():
					fake = gen_model(noise)
				disc_real = disc_model(img_batch.to(DEVICE))
				disc_fake = disc_model(fake)
				disc_real_loss = loss_func(disc_real, torch.ones_like(disc_real))
				disc_fake_loss = loss_func(disc_fake, torch.zeros_like(disc_fake))
				disc_loss = disc_real_loss + disc_fake_loss
				total_disc_loss += disc_loss.item()

				disc_optimiser.zero_grad()
				disc_loss.backward()
				disc_optimiser.step()

				# Train generator

				noise = torch.randn(img_batch.shape[0], GEN_LATENT_DIM, 1, 1, device=DEVICE)
				fake = gen_model(noise)
				disc_fake = disc_model(fake)
				gen_loss = loss_func(disc_fake, torch.ones_like(disc_fake))
				total_gen_loss += gen_loss.item()

				gen_optimiser.zero_grad()
				gen_loss.backward()
				gen_optimiser.step()

			# To track output quality throughout training
			gen_model.eval()
			plot_output(f'./images/epoch_{epoch}.png')

			mean_disc_loss = total_disc_loss / len(train_loader)
			mean_gen_loss = total_gen_loss / len(train_loader)
			prog_bar.finish(f'{mean_disc_loss=:.4f}, {mean_gen_loss=:.4f}')

			torch.save(gen_model.state_dict(), f'./gen_model_epoch_{epoch}.pth')

	gen_model.eval()

	# Test generator on a random noise vector

	plot_output('./images/output.png')

	# Visualise some of the latent space by linearly interpolating between 2 random noise vectors

	noise2 = torch.randn_like(fixed_noise)
	for t in torch.linspace(0, 1, 101):
		noise_interp = noise2 * t + fixed_noise * (1 - t)
		with torch.inference_mode():
			latent_space_test = gen_model(noise_interp)
		plot_image_grid(
			latent_space_test, rows=4, cols=6, padding=4, transform=destandardise_transform, scale_factor=1.5,
			scale_interpolation='cubic', background_rgb=(0, 0, 0), title_rgb=(255, 255, 255),
			title=f'{t:.2f}(vector_1) + {(1 - t):.2f}(vector_2)',
			save_path=f'./images/{t:.2f}.png',
			show=False
		)
