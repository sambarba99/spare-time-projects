"""
SVM class

Author: Sam Barba
Created 13/03/2024
"""

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['figure.figsize'] = (6, 6)


class SVM:
	def __init__(self, learning_rate=0.001, lambda_param=0.001, num_iters=100):
		self.lr = learning_rate
		self.lambda_param = lambda_param
		self.num_iters = num_iters
		self.w = None
		self.b = None

	def fit(self, x, y):
		def plot(iter_num):
			def get_hyperplane_value(x, w, b, offset):
				return (-w[0] * x + b + offset) / w[1]


			plt.cla()
			plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr', alpha=0.5)

			x0_min = x[:, 0].min()
			x0_max = x[:, 0].max()
			x1_min = x[:, 1].min()
			x1_max = x[:, 1].max()
			decision_hyperplane_1 = get_hyperplane_value(x0_min, self.w, self.b, 0)
			decision_hyperplane_2 = get_hyperplane_value(x0_max, self.w, self.b, 0)
			neg_margin_hyperplane_1 = get_hyperplane_value(x0_min, self.w, self.b, -1)
			neg_margin_hyperplane_2 = get_hyperplane_value(x0_max, self.w, self.b, -1)
			pos_margin_hyperplane_1 = get_hyperplane_value(x0_min, self.w, self.b, 1)
			pos_margin_hyperplane_2 = get_hyperplane_value(x0_max, self.w, self.b, 1)

			plt.plot([x0_min, x0_max], [decision_hyperplane_1, decision_hyperplane_2], 'k')
			plt.plot([x0_min, x0_max], [neg_margin_hyperplane_1, neg_margin_hyperplane_2], 'k--')
			plt.plot([x0_min, x0_max], [pos_margin_hyperplane_1, pos_margin_hyperplane_2], 'k--')

			plt.xlim([x0_min - 1, x0_max + 1])
			plt.ylim([x1_min - 1, x1_max + 1])
			plt.title(f'Iter {iter_num}/{self.num_iters}')

			if iter_num == self.num_iters:
				plt.show()
			else:
				plt.draw()
				plt.pause(1e-6)


		y[y == 0] = -1

		self.w = np.zeros(x.shape[1])
		self.b = 0.5  # Arbitrary random value

		for i in range(1, self.num_iters + 1):
			for idx, xi in enumerate(x):
				if y[idx] * (np.dot(xi, self.w) - self.b) >= 1:
					self.w -= self.lr * 2 * self.lambda_param * self.w
				else:
					self.w -= self.lr * (
						2 * self.lambda_param * self.w - np.dot(xi, y[idx])
					)
					self.b -= self.lr * y[idx]
			plot(i)

	def predict(self, x):
		return np.sign(np.dot(x, self.w) - self.b)
