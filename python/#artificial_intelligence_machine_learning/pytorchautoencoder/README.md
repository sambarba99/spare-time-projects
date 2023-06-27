PyTorch implementation of an autoencoder for the MNIST dataset

The encoder reduces the input shape from `(1, 28, 28)` (colour channels, width, height) to `(2,)` - a compression factor of 392. These latent variables can thus be plotted on $xy$ axes:

![](visualised_latent_space.gif)
