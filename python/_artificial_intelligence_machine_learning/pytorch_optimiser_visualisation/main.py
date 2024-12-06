"""
PyTorch optimiser visualisation

Author: Sam Barba
Created 17/11/2024
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
torch.manual_seed(1)


X1_RANGE = -1, 1
X2_RANGE = 0, 1
RANGE_SIZE = 201
LEARNING_RATE = 6e-3
NUM_EPOCHS = 1000


def loss_func(x1, x2):
	"""
	Equivalent 3D surface: z = x^2 + y^2 - xy - x
	"""

	loss = x1 * x1 + x2 * x2 - x1 * x2 - x1

	return loss


def get_optimiser(optimiser_name, **kwargs):
	optimiser_class = getattr(torch.optim, optimiser_name)
	return optimiser_class(**kwargs)


def plot_training_progress(epoch):
	minimum_loss_line = [global_minimum_loss] * epoch

	ax_3d.clear()
	ax_2d.clear()
	ax_loss.clear()

	ax_3d.plot_surface(x1_grid, x2_grid, loss_grid, cmap='coolwarm', antialiased=False)
	contour_plt = ax_2d.contourf(x1_grid, x2_grid, loss_grid, levels=100, cmap='coolwarm')
	if epoch == 0:
		plt.colorbar(contour_plt)

	for optimiser_name, vals in optimisers.items():
		ax_3d.plot(
			*np.array(vals['x_history'][:epoch + 1]).T, vals['loss_history'][:epoch + 1],
			color=vals['plot_colour'], zorder=1
		)
		ax_3d.scatter(
			*vals['x_history'][epoch], vals['loss_history'][epoch], color=vals['plot_colour'],
			s=8, zorder=1
		)

		ax_2d.plot(*np.array(vals['x_history'][:epoch + 1]).T, color=vals['plot_colour'], zorder=1)
		ax_2d.scatter(*vals['x_history'][epoch], color=vals['plot_colour'], s=8, label=optimiser_name, zorder=1)

		ax_loss.plot(range(1, epoch + 1), vals['loss_history'][:epoch], color=vals['plot_colour'], label=optimiser_name)

	ax_3d.scatter(*global_minimum, global_minimum_loss, color='#00c000', marker='*', s=50, zorder=2)
	ax_3d.set_xlabel('$x_1$', fontsize=14)
	ax_3d.set_ylabel('$x_2$', fontsize=14)
	ax_3d.set_zlabel('Loss')

	ax_2d.scatter(*global_minimum, color='#00c000', marker='*', s=50, label='Global minimum', zorder=2)
	ax_2d.set_xlabel('$x_1$', fontsize=14)
	ax_2d.set_ylabel('$x_2$', fontsize=14)
	ax_2d.legend(loc='upper right')

	ax_loss.plot(range(1, epoch + 1), minimum_loss_line, color='tab:green', linestyle='--', label='Minimum loss')
	ax_loss.set_xlabel('Epoch')
	ax_loss.set_ylabel('Loss')
	ax_loss.set_title('Loss per epoch', fontsize=11)
	ax_loss.legend(loc='upper right')

	solution_str = ''
	for optimiser_name, vals in optimisers.items():
		solution_str += f'{optimiser_name} solution:\n'
		solution_str += f"x1,x2 = ({vals['x_history'][epoch][0]:.4f}, {vals['x_history'][epoch][1]:.4f})" \
			f" | loss = {vals['loss_history'][epoch]:.4f}\n"
	solution_str += f'Global minimum:\n' \
		f'x1,x2 = ({global_minimum[0]:.4f}, {global_minimum[1]:.4f})' \
		f' | loss = {global_minimum_loss:.4f}\n'
	plt.figtext(0.63, 0.11, solution_str, fontsize=11)
	num_solution_labels = sum('Global' in str(text) for text in fig.texts)
	if num_solution_labels > 1:
		num_removed = 0
		for text in fig.texts:
			if 'Global' in str(text):
				text.remove()
				num_removed += 1
				if num_removed == num_solution_labels - 1:
					break

	ax_3d.elev = 5 + 30 * epoch / NUM_EPOCHS
	ax_3d.azim = 270 + 45 * epoch / NUM_EPOCHS
	plt.suptitle('Objective function surface plot and contour plot', y=0.95)
	if epoch == NUM_EPOCHS:
		plt.show()
	else:
		plt.draw()
		plt.pause(1e-6)


if __name__ == '__main__':
	fig = plt.figure(figsize=(10, 7))
	ax_3d = plt.axes([0.02, 0.45, 0.44, 0.44], projection='3d', elev=1, azim=270, computed_zorder=False)
	ax_2d = plt.axes([0.56, 0.47, 0.36, 0.4])
	ax_loss = plt.axes([0.11, 0.11, 0.5, 0.24])

	x1_vals = np.linspace(*X1_RANGE, RANGE_SIZE)
	x2_vals = np.linspace(*X2_RANGE, RANGE_SIZE)
	x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
	loss_grid = np.vectorize(loss_func)(x1_grid, x2_grid)

	min_loss_index = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
	global_minimum = x1_grid[min_loss_index], x2_grid[min_loss_index]
	global_minimum_loss = loss_grid[min_loss_index]

	# Start at highest point (max loss)
	max_loss_index = np.unravel_index(np.argmax(loss_grid), loss_grid.shape)
	global_maximum = x1_grid[max_loss_index], x2_grid[max_loss_index]
	global_maximum_loss = loss_grid[max_loss_index]
	x_init = torch.tensor(global_maximum).float()

	optimisers = {
		'Adam': {'x_history': [], 'loss_history': [], 'plot_colour': 'tab:orange'},
		'RMSprop': {'x_history': [], 'loss_history': [], 'plot_colour': 'tab:red'},
		'SGD': {'x_history': [], 'loss_history': [], 'plot_colour': 'tab:purple'}
	}

	for optimiser_name in optimisers:
		x = x_init.clone().requires_grad_()
		optimiser = get_optimiser(optimiser_name, params=[x], lr=LEARNING_RATE)
		optimisers[optimiser_name]['x_history'].append(x_init.clone().numpy())
		optimisers[optimiser_name]['loss_history'].append(global_maximum_loss)

		for _ in range(NUM_EPOCHS):
			loss = loss_func(*x)
			optimiser.zero_grad()
			loss.backward()
			optimiser.step()

			optimisers[optimiser_name]['x_history'].append(x.clone().detach().numpy())
			optimisers[optimiser_name]['loss_history'].append(loss.item())

	for epoch in range(NUM_EPOCHS + 1):
		plot_training_progress(epoch)
