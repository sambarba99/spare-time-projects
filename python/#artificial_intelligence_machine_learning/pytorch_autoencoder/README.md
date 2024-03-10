PyTorch autoencoder demo for the MNIST dataset or for tabular data such as the iris dataset.

In the case of the MNIST autoencoder, the data dimensionality is reduced from `(1, 28, 28)` (colour channels, width, height) to `(2,)` - a compression factor of 392. These latent variables can thus be plotted on $xy$ axes:

![](plots/mnist_2_latent_variables.png)

MNIST autoencoder architecture:

![](plots/mnist_autoencoder_architecture.png)

Example of tabular data compression (iris dataset, 2 latent variables and 3 latent variables):

![](plots/iris_2_latent_variables.png)
![](plots/iris_3_latent_variables.png)
