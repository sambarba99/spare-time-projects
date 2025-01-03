## PyTorch demo of a Generative Denoising Diffusion Probabilistic Model (DDPM)

The model is trained on a forward diffusion process where it must predict the noise added at each step:

<p align="center">
	<img src="images/forward_process.png"/>
</p>

This means the trained model can then be used to iteratively _denoise_ a purely random noise sample (a _reverse_ diffusion process):

<p align="center">
	<img src="images/reverse_process.webp"/>
</p>

The update equation for the reverse diffusion process (`diffusion_controller.generate_images`) is:

$$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\bigg(x_t-\frac{1-\alpha_t}{\sqrt{1-\alpha^T_t}}\epsilon_\theta(x_t,t)\bigg)+\sqrt{\beta_t}z_t$$

Where:
- $x_t$ = noisy image at timestep $t$
- $\alpha_t$ = alpha value (fraction of original image information that remains after each diffusion step) at $t$
- $\alpha^T_t$ = cumulative product of alphas up to $T$
- $\epsilon_\theta(x_t,t)$ = predicted noise in $x_t$ by model $\epsilon$ (with parameters $\theta$)
- $\beta_t$ = beta value (noise variance added at each diffusion step) at $t$
- $z_t$ = noise term sampled from a Gaussian distribution, added for stochasticity.

Model architecture:

<p align="center">
	<img src="images/model_architecture.png"/>
</p>

Sources:
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) (Ho, Jain, Abbeel 2020)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) (Vaswani et. al. 2023)
- [CelebA-HQ resized (256x256)](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256) (Kaggle dataset)
