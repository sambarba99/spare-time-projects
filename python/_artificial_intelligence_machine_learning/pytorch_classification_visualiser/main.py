"""
Visualising the decision boundary of a PyTorch classification neural net

Author: Sam Barba
Created 04/11/2022
"""

import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
import torch
from torch import nn


plt.rcParams['figure.figsize'] = (6, 6)

N_EPOCHS = 1000
N_SAMPLES = 1000
LAYER_SIZE = 32

x = y = n_classes = None
model = nn.Sequential()


def make_spiral_classes():
	clockwise_class, anticlockwise_class = [], []

	for i in range(N_SAMPLES // 2):
		theta = i / 180 * np.pi
		r = i / 500
		x1 = r * np.cos(theta)
		x2 = r * np.sin(theta)
		clockwise_class.append([x1, x2])
		anticlockwise_class.append([-x1, -x2])

	return np.array(clockwise_class), np.array(anticlockwise_class)


def gen_data(mode):
	global x, y, n_classes

	match mode:
		case 'clusters':
			x, y = make_blobs(n_samples=N_SAMPLES, centers=5, cluster_std=2)
		case 'circles':
			x, y = make_circles(n_samples=N_SAMPLES, noise=0.15, factor=0.5)
		case 'moons':
			x, y = make_moons(n_samples=N_SAMPLES, noise=0.15)
			x[:, 1] *= 1.7  # Stretch a bit in the geometric y direction
		case 'spirals':
			class1, class2 = make_spiral_classes()
			x = np.vstack((class1, class2))
			x += np.random.normal(scale=0.06, size=x.shape)
			y1 = np.zeros(N_SAMPLES // 2)
			y2 = np.ones(N_SAMPLES // 2)
			y = np.concatenate((y1, y2))
		case _:
			raise ValueError(f'Bad data gen mode: {mode}')

	n_classes = len(np.unique(y))
	if n_classes > 2:
		y = np.eye(n_classes)[y]  # One-hot encode

	x, y = torch.tensor(x).float(), torch.tensor(y).float()

	build_model()
	plot_decision_boundary()
	plt.title('Start (random weights)')
	plt.show()


def build_model():
	global model

	model = nn.Sequential(
		nn.Linear(in_features=2, out_features=LAYER_SIZE),
		nn.Tanh(),
		nn.Linear(LAYER_SIZE, LAYER_SIZE),
		nn.Tanh(),
		nn.Linear(LAYER_SIZE, 1 if n_classes == 2 else n_classes),
		nn.Sigmoid() if n_classes == 2 else nn.Softmax(dim=-1)
	)


def train_model():
	build_model()
	loss_func = nn.BCELoss() if n_classes == 2 else nn.CrossEntropyLoss()
	optimiser = torch.optim.Adam(model.parameters())  # LR = 1e-3

	for epoch in range(1, N_EPOCHS + 1):
		y_probs = model(x).squeeze()
		loss = loss_func(y_probs, y)

		optimiser.zero_grad()
		loss.backward()
		optimiser.step()

		if epoch % 10 == 0:
			plot_decision_boundary()
			plt.title(f'Epoch {epoch}/{N_EPOCHS}')
			plt.draw()
			plt.pause(0.01)

	plot_decision_boundary()
	plt.title(f'Epoch {N_EPOCHS}/{N_EPOCHS}')
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
	y_probs = model(x_to_pred).squeeze()

	if n_classes == 2:  # Binary
		y_pred = y_probs.round()
	else:  # Multiclass
		y_pred = y_probs.argmax(dim=1)

	# Reshape and plot
	plt.cla()
	y_flat = y if y.dim() == 1 else y.argmax(dim=1)
	plt.scatter(x[:, 0], x[:, 1], c=y_flat, cmap='jet', alpha=0.7)
	y_pred = y_pred.reshape(xx.shape).detach().numpy()
	plt.imshow(
		y_pred, interpolation='nearest', cmap='jet', alpha=0.5, aspect='auto', origin='lower',
		extent=(xx.min(), xx.max(), yy.min(), yy.max())
	)


if __name__ == '__main__':
	root = tk.Tk()
	root.title('Neural net boundary visualiser')
	root.config(width=350, height=240, background='#101010')
	root.resizable(False, False)

	gen_data_lbl = tk.Label(root, text='1. Generate data',
		font='consolas', background='#101010', foreground='white')
	train_model_lbl = tk.Label(root, text='2. Train model',
		font='consolas', background='#101010', foreground='white')

	btn_clusters = tk.Button(root, text='Clusters', font='consolas', command=lambda: gen_data('clusters'))
	btn_circles = tk.Button(root, text='Circles', font='consolas', command=lambda: gen_data('circles'))
	btn_moons = tk.Button(root, text='Moons', font='consolas', command=lambda: gen_data('moons'))
	btn_spirals = tk.Button(root, text='Spirals', font='consolas', command=lambda: gen_data('spirals'))
	btn_train = tk.Button(root, text='Train', font='consolas', command=train_model)

	gen_data_lbl.place(width=298, height=36, relx=0.5, y=36, anchor='center')
	train_model_lbl.place(width=298, height=36, relx=0.5, y=156, anchor='center')
	btn_clusters.place(width=105, height=36, relx=0.344, y=72, anchor='center')
	btn_circles.place(width=105, height=36, relx=0.656, y=72, anchor='center')
	btn_moons.place(width=105, height=36, relx=0.344, y=113, anchor='center')
	btn_spirals.place(width=105, height=36, relx=0.656, y=113, anchor='center')
	btn_train.place(width=105, height=36, relx=0.5, y=190, anchor='center')

	gen_data('clusters')

	root.mainloop()
