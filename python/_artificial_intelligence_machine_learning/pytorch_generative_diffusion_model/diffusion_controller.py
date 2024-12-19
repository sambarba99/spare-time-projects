"""
Class for controlling the forward/reverse diffusion processes

Author: Sam Barba
Created 28/05/2024
"""

import torch
from tqdm import tqdm

from _utils.plotting import plot_image_grid


class DiffusionController:
	def __init__(self, *, num_timesteps, beta_min, beta_max, device):
		self.T = num_timesteps
		self.betas = torch.linspace(beta_min, beta_max, self.T, device=device)  # Linear noise schedule
		self.alphas = 1 - self.betas
		self.alphas_cp = torch.cumprod(self.alphas, dim=0)
		self.sqrt_alphas_cp = self.alphas_cp.sqrt()
		self.sqrt_1_minus_alphas_cp = (1 - self.alphas_cp).sqrt()
		self.device = device

	def add_noise(self, images, timesteps):
		sqrt_alphas_cp_t = self.sqrt_alphas_cp[timesteps].view(-1, 1, 1, 1)
		sqrt_1_minus_alphas_cp_t = self.sqrt_1_minus_alphas_cp[timesteps].view(-1, 1, 1, 1)
		noise = torch.randn_like(images)
		noisy_images = sqrt_alphas_cp_t * images + sqrt_1_minus_alphas_cp_t * noise

		return noisy_images, noise

	def generate_images(self, model, img_size, num_images=24):
		"""Starting from pure noise, generate images via a reverse diffusion process"""

		images = torch.randn(num_images, 3, img_size, img_size, device=self.device)

		plot_image_grid(
			images, rows=4, cols=6, padding=4, scale_factor=1.5, scale_interpolation='cubic',
			background_rgb=(0, 0, 0), title_rgb=(255, 255, 255),
			title=f'Reverse diffusion process (t={self.T}/{self.T})',
			save_path=f'./images/reverse_diffusion_step_{self.T}.png',
			show=False
		)

		for t in tqdm(reversed(range(self.T)), desc='Iterating over reversed timesteps', ascii=True, total=self.T):
			# Predict noise
			t_tensor = torch.full((num_images,), t, device=self.device)
			noise_pred = model(images, t_tensor)

			# Partial denoising
			alpha_t = self.alphas[t]
			alpha_cp_t = self.alphas_cp[t]
			images = (1 / alpha_t.sqrt()) * (images - (1 - alpha_t) / (1 - alpha_cp_t).sqrt() * noise_pred)

			if t > 0:
				# Add noise for all t > 0: during training, the model learns how to gradually noise images
				# (the forward diffusion process). In the reverse process, at each timestep, the model predicts
				# the current noise in the image. Based on this, it is then slightly denoised. To maintain the
				# correct distribution and prevent images collapsing to a single mode, or becoming overly
				# deterministic, additional noise is added back at each step. This stochasticity is crucial for
				# proper sampling from the learned distribution.

				beta_t = self.betas[t]
				noise = torch.randn_like(images)
				images += beta_t.sqrt() * noise

			plot_image_grid(
				images, rows=4, cols=6, padding=4, scale_factor=1.5, scale_interpolation='cubic',
				background_rgb=(0, 0, 0), title_rgb=(255, 255, 255),
				title=f'Reverse diffusion process (t={t}/{self.T})',
				save_path=f'./images/reverse_diffusion_step_{t:0>4}.png',
				show=False
			)

		plot_image_grid(
			images, rows=4, cols=6, padding=4, scale_factor=1.5, scale_interpolation='cubic',
			background_rgb=(0, 0, 0), title_rgb=(255, 255, 255),
			title='Model test on random noise'
		)
