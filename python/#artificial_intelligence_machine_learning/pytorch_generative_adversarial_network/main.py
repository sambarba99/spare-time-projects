"""
PyTorch demo of a Deep Convolutional Generative Adversarial Network (DCGAN)

Author: Sam Barba
Created 01/07/2023
"""

import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models import Discriminator, Generator
from my_dataset import MyDataset


plt.rcParams['figure.figsize'] = (6, 6)
torch.manual_seed(1)

IMG_SIZE = 64
DISC_NOISE_STRENGTH = 0.05
GEN_LATENT_DIM = 128
N_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
OPTIM_BETAS = (0.5, 0.999)


def create_train_loader():
	transform = transforms.Compose([
		transforms.Resize(IMG_SIZE),
		# transforms.CenterCrop(IMG_SIZE),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalise to [-1,1]
	])

	dataset = MyDataset('C:/Users/Sam/Desktop/Projects/datasets/UTKFace', transform)
	train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

	return train_loader


def plot_gen_output(gen_images, title, save_path=None):
	_, axes = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
	plt.gcf().set_facecolor('black')
	plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.05, wspace=0.05)
	for idx, ax in enumerate(axes.flatten()):
		gen_img = (gen_images[idx] + 1) * 127.5  # De-normalise
		gen_img = gen_img.type(torch.uint8).permute(1, 2, 0)
		ax.imshow(gen_img)
		ax.axis('off')
	plt.suptitle(title, y=0.95, color='white')
	if save_path:
		plt.savefig(save_path)
	else:
		plt.show()
	plt.close()


if __name__ == '__main__':
	disc_model = Discriminator(noise_strength=DISC_NOISE_STRENGTH)
	gen_model = Generator(latent_dim=GEN_LATENT_DIM)
	print(f'\nModels:\n\n{disc_model}\n{gen_model}')

	fixed_noise = torch.randn(25, GEN_LATENT_DIM, 1, 1)

	if os.path.exists('./gen_model.pth'):
		gen_model.load_state_dict(torch.load('./gen_model.pth'))
	else:
		print('\n----- TRAINING -----\n')

		train_loader = create_train_loader()
		loss_func = torch.nn.BCELoss()
		disc_optimiser = torch.optim.Adam(disc_model.parameters(), lr=LEARNING_RATE, betas=OPTIM_BETAS)
		gen_optimiser = torch.optim.Adam(gen_model.parameters(), lr=LEARNING_RATE, betas=OPTIM_BETAS)

		gen_model.eval()
		with torch.inference_mode():
			fake_images_test = gen_model(fixed_noise)
		plot_gen_output(fake_images_test, 'Start', './images/0_start.png')

		for epoch in range(1, N_EPOCHS + 1):
			for batch_idx, img_batch in enumerate(train_loader, start=1):
				gen_model.train()
				noise = torch.randn(len(img_batch), GEN_LATENT_DIM, 1, 1)
				fake = gen_model(noise)

				# Train discriminator

				disc_real = disc_model(img_batch)
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

				gen_optimiser.zero_grad()
				gen_loss.backward()
				gen_optimiser.step()

				# Plot generation progress

				gen_model.eval()
				with torch.inference_mode():
					fake_images_test = gen_model(fixed_noise)
				title = f'Epoch {epoch}/{N_EPOCHS}, iteration {batch_idx}/{len(train_loader)}'
				save_path = f'./images/ep_{epoch:03}_iter_{batch_idx:03}.png'
				plot_gen_output(fake_images_test, title, save_path)

				if batch_idx % 10 == 0:
					print(f'{title.replace("iteration", "batch")}  |  '
						f'Disc real loss: {disc_real_loss.item():.4f}  |  '
						f'Disc fake loss: {disc_fake_loss.item():.4f}  |  '
						f'Gen loss: {gen_loss.item():.4f}')

		torch.save(gen_model.state_dict(), './gen_model.pth')

	gen_model.eval()
	with torch.inference_mode():
		fake_images_test = gen_model(fixed_noise)
	plot_gen_output(fake_images_test, 'Generator test')
