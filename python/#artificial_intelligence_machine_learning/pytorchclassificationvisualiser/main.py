"""
PyTorch classification neural net visualiser

Author: Sam Barba
Created 04/11/2022
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
import tkinter as tk
import torch
from torch import nn

plt.rcParams['figure.figsize'] = (6, 6)

x = y = n_classes = lr = None
model = nn.Sequential()

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def make_spiral_classes():
	clockwise_class, anticlockwise_class = [], []

	for i in range(250):
		theta = i / 88 * np.pi
		r = i / 250
		x1 = r * np.cos(theta)
		x2 = r * np.sin(theta)
		clockwise_class.append([x1, x2])
		anticlockwise_class.append([-x1, -x2])

	return np.array(clockwise_class), np.array(anticlockwise_class)

def gen_data(mode):
	global x, y, n_classes, lr

	match mode:
		case '3 blobs':
			x, y = make_blobs(n_samples=500, centers=3, cluster_std=2)
			lr = 3e-4
		case '4 blobs':
			x, y = make_blobs(n_samples=500, centers=4, cluster_std=2)
			lr = 4e-4
		case '5 blobs':
			x, y = make_blobs(n_samples=500, centers=5, cluster_std=2)
			lr = 5e-4
		case 'circles':
			x, y = make_circles(n_samples=500, noise=0.15, factor=0.5)
			lr = 8e-5
		case 'moons':
			x, y = make_moons(n_samples=500, noise=0.15)
			x[:, 1] *= 1.7  # Stretch a bit in the y (geometric) direction
			lr = 3e-4
		case 'spirals':
			class1, class2 = make_spiral_classes()
			x = np.vstack((class1, class2))
			x += np.random.normal(scale=0.05, size=x.shape)
			y1 = np.zeros(class1.shape[0])
			y2 = np.ones(class2.shape[0])
			y = np.concatenate((y1, y2))
			lr = 6e-4
		case _:
			raise ValueError(f'Bad data gen mode: {mode}')

	n_classes = len(np.unique(y))
	if n_classes > 2:
		y = np.eye(n_classes)[y]  # One-hot encode

	# Put data and model to CPU (works better with numpy and matplotlib)
	x = torch.from_numpy(x).float().to('cpu')
	y = torch.from_numpy(y).float().to('cpu')

	build_model()
	plot_decision_boundary()
	plt.title('Start (random weights)')
	plt.show()

def build_model():
	global model

	model = nn.Sequential(
		nn.Linear(in_features=2, out_features=64),
		nn.ReLU(),
		nn.Linear(64, 64),
		nn.ReLU(),
		nn.Linear(64, 1 if n_classes == 2 else n_classes),
		nn.Sigmoid() if n_classes == 2 else nn.Softmax(dim=1)
	)
	model.to('cpu')

def train_model():
	loss_func = nn.BCELoss() if n_classes == 2 else nn.CrossEntropyLoss()
	optimiser = torch.optim.Adam(model.parameters(), lr=lr)

	for epoch in range(1000):
		model.train()
		y_probs = model(x).squeeze()
		loss = loss_func(y_probs, y)

		optimiser.zero_grad()
		loss.backward()
		optimiser.step()

		if epoch % 20 == 0:
			plot_decision_boundary()
			plt.title(f'Epoch {epoch}/1000')
			plt.show(block=False)
			plt.pause(0.01)

	plot_decision_boundary()
	plt.title('Epoch 1000/1000')
	plt.show()

def plot_decision_boundary():
	# Set up prediction boundaries and grid
	x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
	y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
	xx, yy = np.meshgrid(
		np.linspace(x_min, x_max, 250),
		np.linspace(y_min, y_max, 250)
	)

	# Make features
	x_to_pred = torch.from_numpy(np.stack((xx.ravel(), yy.ravel()), axis=1)).float()

	# Make predictions
	model.eval()
	with torch.inference_mode():
		y_probs = model(x_to_pred).squeeze()

	if n_classes == 2:  # Binary
		y_pred = y_probs.round()
	else:  # Multiclass
		y_pred = y_probs.argmax(dim=1)

	# Reshape and plot
	y_pred = y_pred.reshape(xx.shape).detach().numpy()
	y_flat = y if y.dim() == 1 else y.argmax(dim=1)
	plt.cla()
	plt.contourf(xx, yy, y_pred, alpha=0.4, cmap=plt.cm.jet)
	plt.scatter(x[:, 0], x[:, 1], c=y_flat, cmap=plt.cm.jet)
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.axis('scaled')

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	root = tk.Tk()
	root.title('PyTorch classification neural net visualiser')
	root.config(width=400, height=240, bg='#000024')
	root.resizable(False, False)

	gen_data_lbl = tk.Label(root, text='1. Generate data',
		font='consolas', bg='#000024', fg='white')
	train_model_lbl = tk.Label(root, text='2. Train model',
		font='consolas', bg='#000024', fg='white')

	btn_3_blobs = tk.Button(root, text='3 blobs', font='consolas',
		command=lambda: gen_data('3 blobs'))
	btn_4_blobs = tk.Button(root, text='4 blobs', font='consolas',
		command=lambda: gen_data('4 blobs'))
	btn_5_blobs = tk.Button(root, text='5 blobs', font='consolas',
		command=lambda: gen_data('5 blobs'))
	btn_circles = tk.Button(root, text='Circles', font='consolas',
		command=lambda: gen_data('circles'))
	btn_moons = tk.Button(root, text='Moons', font='consolas',
		command=lambda: gen_data('moons'))
	btn_spirals = tk.Button(root, text='Spirals', font='consolas',
		command=lambda: gen_data('spirals'))
	btn_train = tk.Button(root, text='Train', font='consolas',
		command=train_model)

	gen_data_lbl.place(relwidth=0.85, relheight=0.15, relx=0.5, rely=0.15, anchor='center')
	train_model_lbl.place(relwidth=0.85, relheight=0.15, relx=0.5, rely=0.65, anchor='center')
	btn_3_blobs.place(relwidth=0.25, relheight=0.15, relx=0.24, rely=0.3, anchor='center')
	btn_4_blobs.place(relwidth=0.25, relheight=0.15, relx=0.5, rely=0.3, anchor='center')
	btn_5_blobs.place(relwidth=0.25, relheight=0.15, relx=0.76, rely=0.3, anchor='center')
	btn_circles.place(relwidth=0.25, relheight=0.15, relx=0.24, rely=0.47, anchor='center')
	btn_moons.place(relwidth=0.25, relheight=0.15, relx=0.5, rely=0.47, anchor='center')
	btn_spirals.place(relwidth=0.25, relheight=0.15, relx=0.76, rely=0.47, anchor='center')
	btn_train.place(relwidth=0.5, relheight=0.15, relx=0.5, rely=0.79, anchor='center')

	gen_data('3 blobs')

	root.mainloop()
