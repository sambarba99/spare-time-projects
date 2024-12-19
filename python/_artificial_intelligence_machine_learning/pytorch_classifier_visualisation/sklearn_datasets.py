"""
Visualising the decision boundary of a PyTorch classification neural net

Author: Sam Barba
Created 04/11/2022
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
import torch
from torch import nn


plt.rcParams['figure.figsize'] = (7, 7)

NUM_EPOCHS = 1000
NUM_SAMPLES = 1000
LAYER_SIZE = 32

x = y = num_classes = None
model = nn.Sequential()


def make_spirals():
	clockwise_class, anticlockwise_class = [], []

	for i in range(NUM_SAMPLES // 2):
		theta = i / 180 * np.pi
		r = i / 500
		x_coord = r * np.cos(theta)
		y_coord = r * np.sin(theta)
		clockwise_class.append([x_coord, y_coord])
		anticlockwise_class.append([-x_coord, -y_coord])

	class1, class2 = np.array(clockwise_class), np.array(anticlockwise_class)
	x = np.vstack((class1, class2))
	x += np.random.normal(scale=0.06, size=x.shape)
	y1 = np.zeros(NUM_SAMPLES // 2)
	y2 = np.ones(NUM_SAMPLES // 2)
	y = np.concatenate((y1, y2))

	return x, y


def gen_data(mode):
	global x, y, num_classes

	match mode:
		case 'clusters':
			x, y = make_blobs(n_samples=NUM_SAMPLES, centers=5, cluster_std=2)
		case 'circles':
			x, y = make_circles(n_samples=NUM_SAMPLES, noise=0.15, factor=0.5)
		case 'moons':
			x, y = make_moons(n_samples=NUM_SAMPLES, noise=0.15)
			x[:, 1] *= 1.7  # Stretch a bit in the geometric y direction
		case _:
			x, y = make_spirals()

	num_classes = len(np.unique(y))

	x = torch.tensor(x).float()
	y = torch.tensor(y).float() if num_classes == 2 else torch.tensor(y).long()

	build_model()
	plot_decision_boundary()
	ax_classification.set_title('Start (random weights)')
	plt.show()


def build_model():
	global model

	model = nn.Sequential(
		nn.Linear(2, LAYER_SIZE),
		nn.Tanh(),
		nn.Linear(LAYER_SIZE, LAYER_SIZE),
		nn.Tanh(),
		nn.Linear(LAYER_SIZE, 1 if num_classes == 2 else num_classes)
	).cpu()


def build_and_train_model(*_):
	build_model()
	loss_func = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
	optimiser = torch.optim.Adam(model.parameters())  # LR = 1e-3

	for epoch in range(1, NUM_EPOCHS + 1):
		y_logits = model(x).squeeze()
		loss = loss_func(y_logits, y)

		optimiser.zero_grad()
		loss.backward()
		optimiser.step()

		if epoch % 10 == 0:
			plot_decision_boundary()
			ax_classification.set_title(f'Epoch {epoch}/{NUM_EPOCHS}')
			plt.pause(0.01)

	plot_decision_boundary()
	ax_classification.set_title(f'Epoch {NUM_EPOCHS}/{NUM_EPOCHS}')
	plt.show()


def plot_decision_boundary():
	# Set up prediction boundaries and grid
	x_min, x_max = x[:, 0].min(), x[:, 0].max()
	y_min, y_max = x[:, 1].min(), x[:, 1].max()
	xx, yy = np.meshgrid(
		np.linspace(x_min - 0.1, x_max + 0.1, 250),
		np.linspace(y_min - 0.1, y_max + 0.1, 250)
	)
	mesh_coords = np.column_stack((xx.flatten(), yy.flatten()))

	# Make features
	x_to_pred = torch.tensor(mesh_coords).float()

	# Make predictions
	# (no need to use eval() or inference_mode() as model doesn't have dropout or batch norm)
	y_logits = model(x_to_pred).squeeze()

	if num_classes == 2:  # Binary
		y_probs = torch.sigmoid(y_logits)
		y_pred = y_probs.round()
	else:  # Multiclass
		y_pred = y_logits.argmax(dim=1)

	# Reshape and plot
	ax_classification.clear()
	y_flat = y if y.dim() == 1 else y.argmax(dim=1)
	ax_classification.scatter(x[:, 0], x[:, 1], c=y_flat, cmap='jet', alpha=0.7)
	y_pred = y_pred.reshape(xx.shape).detach().numpy()
	ax_classification.imshow(
		y_pred, interpolation='nearest', cmap='jet', alpha=0.5, aspect='auto', origin='lower',
		extent=(xx.min(), xx.max(), yy.min(), yy.max())
	)


if __name__ == '__main__':
	ax_classification = plt.axes([0.18, 0.28, 0.64, 0.64])
	ax_gen_clusters = plt.axes([0.29, 0.16, 0.2, 0.05])
	ax_gen_circles = plt.axes([0.5, 0.16, 0.2, 0.05])
	ax_gen_moons = plt.axes([0.29, 0.1, 0.2, 0.05])
	ax_gen_spirals = plt.axes([0.5, 0.1, 0.2, 0.05])
	ax_train_model = plt.axes([0.29, 0.04, 0.41, 0.05])

	gen_clusters_btn = Button(ax_gen_clusters, 'Generate clusters')
	gen_circles_btn = Button(ax_gen_circles, 'Generate circles')
	gen_moons_btn = Button(ax_gen_moons, 'Generate moons')
	gen_spirals_btn = Button(ax_gen_spirals, 'Generate spirals')
	train_model_btn = Button(ax_train_model, 'Train model')

	gen_clusters_btn.on_clicked(lambda _: gen_data('clusters'))
	gen_circles_btn.on_clicked(lambda _: gen_data('circles'))
	gen_moons_btn.on_clicked(lambda _: gen_data('moons'))
	gen_spirals_btn.on_clicked(lambda _: gen_data('spirals'))
	train_model_btn.on_clicked(build_and_train_model)

	gen_data('clusters')
