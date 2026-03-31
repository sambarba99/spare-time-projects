"""
SVM class

Author: Sam Barba
Created 13/03/2024
"""

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['figure.figsize'] = (6, 6)


def hyperplane(x, w, b, offset):
	return (-w[0] * x - b + offset) / w[1]


class SVM:
	def __init__(self, learning_rate=1e-3, lambda_param=1e-3, num_iters=200, animate=True):
		self.lr = learning_rate
		self.lambda_param = lambda_param
		self.num_iters = num_iters
		self.animate = animate
		self.mean = None
		self.std = None
		self.w = None
		self.b = None

	def plot(self, iter_num, x, y):
		plt.cla()
		plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr', alpha=0.6)

		# Ensure axes are centered and equal
		x0_spread = x[:, 0].max() - x[:, 0].min()
		x1_spread = x[:, 1].max() - x[:, 1].min()
		spread = max(x0_spread, x1_spread) + 2
		x0_min = x[:, 0].mean() - spread / 2
		x0_max = x[:, 0].mean() + spread / 2
		x1_min = x[:, 1].mean() - spread / 2
		x1_max = x[:, 1].mean() + spread / 2

		# De-standardise for plotting
		w_orig = self.w / self.std
		b_orig = self.b - np.sum((self.w * self.mean) / self.std)

		# Decision boundary + margins
		plt.plot(
			[x0_min, x0_max],
			[hyperplane(x0_min, w_orig, b_orig, 0), hyperplane(x0_max, w_orig, b_orig, 0)],
			'k'
		)
		plt.plot(
			[x0_min, x0_max],
			[hyperplane(x0_min, w_orig, b_orig, 1), hyperplane(x0_max, w_orig, b_orig, 1)],
			'k--'
		)
		plt.plot(
			[x0_min, x0_max],
			[hyperplane(x0_min, w_orig, b_orig, -1), hyperplane(x0_max, w_orig, b_orig, -1)],
			'k--'
		)

		# Colour each side of decision boundary
		xx, yy = np.meshgrid(
			np.linspace(x0_min, x0_max, 250),
			np.linspace(x1_min, x1_max, 250)
		)
		mesh_coords = np.column_stack((xx.flatten(), yy.flatten()))
		z = self.predict(mesh_coords).reshape(xx.shape)
		plt.contourf(xx, yy, z, alpha=0.3, cmap='bwr')

		plt.xlim(x0_min, x0_max)
		plt.ylim(x1_min, x1_max)
		plt.title(f'Iter {iter_num}/{self.num_iters}'
				f'\nw = [{self.w[0]:.3f}, {self.w[1]:.3f}], b = {self.b:.3f}')

		# plt.savefig(f'./{iter_num:0>3}.png')
		if iter_num == self.num_iters:
			plt.show()
		else:
			plt.draw()
			plt.pause(1e-6)

	def fit(self, x, y):
		# Standardise x
		self.mean = x.mean(axis=0)
		self.std = x.std(axis=0)
		x_standardised = (x - self.mean) / self.std

		y = np.where(y <= 0, -1, 1)

		self.w = np.zeros(x.shape[1])
		self.b = 0

		for i in range(1, self.num_iters + 1):
			# Decision function: f(X) = wX + b
			scores = x_standardised.dot(self.w) + self.b

			# Condition for hinge loss: y * f(X)
			margins = y * scores

			misclassified = margins < 1

			# Gradient for weights
			dw = 2 * self.lambda_param * self.w - x_standardised[misclassified].T.dot(y[misclassified])

			# Gradient for bias
			db = -y[misclassified].sum()

			# Update params
			self.w -= self.lr * dw
			self.b -= self.lr * db

			if self.animate:
				self.plot(i, x, y)

		if not self.animate:
			# Plot only at the end
			self.plot(self.num_iters, x, y)

	def predict(self, x):
		x = (x - self.mean) / self.std

		return np.where(x.dot(self.w) + self.b >= 0, 1, -1)
