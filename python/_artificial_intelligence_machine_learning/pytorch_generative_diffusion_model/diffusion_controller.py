"""
Class for controlling the forward/reverse diffusion processes

Author: Sam Barba
Created 28/05/2024
"""

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


class DiffusionController:
	def __init__(self, *, num_timesteps, beta_min, beta_max, device):
		self.T = num_timesteps
		self.betas = torch.linspace(beta_min, beta_max, self.T, device=device)  # Linear noise schedule
		self.alphas = 1 - self.betas
		self.alphas_cp = torch.cumprod(self.alphas, dim=0)
		self.sqrt_alphas_cp = torch.sqrt(self.alphas_cp)
		self.sqrt_1_minus_alphas_cp = torch.sqrt(1 - self.alphas_cp)
		self.device = device

	def add_noise(self, images, timesteps):
		sqrt_alphas_cp_t = self.sqrt_alphas_cp[timesteps].view(-1, 1, 1, 1)
		sqrt_1_minus_alphas_cp_t = self.sqrt_1_minus_alphas_cp[timesteps].view(-1, 1, 1, 1)
		noise = torch.randn_like(images)
		noisy_images = sqrt_alphas_cp_t * images + sqrt_1_minus_alphas_cp_t * noise

		return noisy_images, noise

	def generate_images(self, model, num_images, img_size):
		"""Starting from pure noise, generate images via a reverse diffusion process"""

		images = torch.randn(num_images, 3, img_size, img_size, device=self.device)

		self.plot_images(
			torch.clamp(images, -1, 1),
			f'Reverse diffusion process (t={self.T}/{self.T})',
			f'./images/reverse_diffusion_step_{self.T}.png'
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
				# Add noise for all t > 0: during training, the model learns how to gradually noise images (the forward
				# diffusion process). In the reverse process, at each timestep, the model predicts the current noise in
				# the image. Based on this, it is then slightly denoised. To maintain the correct distribution and prevent
				# images collapsing to a single mode, or becoming overly deterministic, additional noise is added back at
				# each step. This stochasticity is crucial for proper sampling from the learned distribution.

				beta_t = self.betas[t]
				noise = torch.randn_like(images)
				images += beta_t.sqrt() * noise

			self.plot_images(
				torch.clamp(images, -1, 1),
				f'Reverse diffusion process (t={t}/{self.T})',
				f'./images/reverse_diffusion_step_{t:0>4}.png'
			)

		self.plot_images(torch.clamp(images, -1, 1), 'Model test on random noise')

	def plot_images(self, images, title, save_path=None):
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
