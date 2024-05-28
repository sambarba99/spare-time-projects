"""
Minimising the distance from a point to multiple 2D or 3D lines

Author: Sam Barba
Created 26/11/2023
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


plt.rcParams['figure.figsize'] = (8, 8)

DIM = 2  # 2 or 3 for 2D or 3D
NUM_LINES = 4

scipy_solution = None


class Line:
	"""
	A 2D or 3D line defined by 2 points (x1, y1[, z1]) and (x2, y2[, z2]).
	In the constructor, 'points' contains (x1, y1[, z1], x2, y2[, z2]).
	"""

	def __init__(self, *points):
		assert len(points) == DIM * 2
		self.point1 = np.array(points[:DIM])
		self.point2 = np.array(points[DIM:])
		self.direction = self.point2 - self.point1

	def distance(self, point):
		vector_to_point = point - self.point1
		projection = np.dot(vector_to_point, self.direction) / np.dot(self.direction, self.direction)
		closest_point_on_line = self.point1 + projection * self.direction
		distance = np.linalg.norm(point - closest_point_on_line)
		return closest_point_on_line, distance


def plot_lines_and_nearest_point(title, ax, lines, nearest_point, trajectory):
	ax.clear()

	ax.scatter(*np.array(trajectory[0]).T, color='red', marker='x', label='SGD initial guess', zorder=2)
	ax.plot(*np.array(trajectory).T, color='black', linestyle=':', linewidth=1, label='SGD learning curve', zorder=1)
	ax.scatter(*nearest_point, color='black', label='Optimal point P', zorder=6)

	for line in lines:
		line_points = np.array([line.point1, line.point2])
		nearest_point_on_line, dist = line.distance(nearest_point)
		perpendicular_line_coords = zip(nearest_point, nearest_point_on_line)
		ax.plot(*line_points.T, zorder=3)
		ax.scatter(*nearest_point_on_line, label=f'Nearest point to P (dist = {dist:.3f})', zorder=5)
		ax.plot(*perpendicular_line_coords, color='black', linestyle='--', linewidth=1, zorder=4)

	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	if DIM == 3:
		ax.set_zlim(0, 1)
		ax.set_zlabel('Z')

	ax.legend()
	plt.title(title)
	plt.draw()
	plt.pause(1e-9)


def generate_lines(n):
	lines = []
	for _ in range(n):
		if DIM == 2:
			point1_side, point2_side = np.random.choice(list('NESW'), size=2, replace=False)
			match point1_side:
				case 'N': x1, y1 = np.random.rand(), 1
				case 'E': x1, y1 = 1, np.random.rand()
				case 'S': x1, y1 = np.random.rand(), 0
				case _: x1, y1 = 0, np.random.rand()
			match point2_side:
				case 'N': x2, y2 = np.random.rand(), 1
				case 'E': x2, y2 = 1, np.random.rand()
				case 'S': x2, y2 = np.random.rand(), 0
				case _: x2, y2 = 0, np.random.rand()
			lines.append(Line(x1, y1, x2, y2))
		else:
			# Assume looking straight on, X goes from east-west, Y from front-back, Z from north-south
			point1_side, point2_side = np.random.choice(list('NESWFB'), size=2, replace=False)  # NESW + front + back
			match point1_side:
				case 'N': x1, y1, z1 = np.random.rand(), np.random.rand(), 1
				case 'E': x1, y1, z1 = 0, np.random.rand(), np.random.rand()
				case 'S': x1, y1, z1 = np.random.rand(), np.random.rand(), 0
				case 'W': x1, y1, z1 = 1, np.random.rand(), np.random.rand()
				case 'F': x1, y1, z1 = np.random.rand(), 0, np.random.rand()
				case _: x1, y1, z1 = np.random.rand(), 1, np.random.rand()
			match point2_side:
				case 'N': x2, y2, z2 = np.random.rand(), np.random.rand(), 1
				case 'E': x2, y2, z2 = 0, np.random.rand(), np.random.rand()
				case 'S': x2, y2, z2 = np.random.rand(), np.random.rand(), 0
				case 'W': x2, y2, z2 = 1, np.random.rand(), np.random.rand()
				case 'F': x2, y2, z2 = np.random.rand(), 0, np.random.rand()
				case _: x2, y2, z2 = np.random.rand(), 1, np.random.rand()
			lines.append(Line(x1, y1, z1, x2, y2, z2))

	return lines


def objective_function(point, lines):
	total_distance_squared = 0
	for line in lines:
		_, distance = line.distance(point)
		total_distance_squared += distance ** 2
	return total_distance_squared


def solve_with_sgd(point, lines, learning_rate=0.02, threshold=0.001, max_iterations=1000):
	trajectory = [point.copy()]

	for i in range(max_iterations):
		gradients = np.zeros(DIM)
		for line in lines:
			nearest_line_point, _ = line.distance(point)
			gradients += 2 * (point - nearest_line_point)
		point -= learning_rate * gradients
		trajectory.append(point.copy())

		sum_grads = np.abs(gradients).sum()
		point_format = ', '.join(f'{i:.3f}' for i in point)
		grad_format = ', '.join(f'{i:.3f}' for i in gradients)
		plot_lines_and_nearest_point(
			f'Iteration: {i + 1}'
			f'\nGradients: ({grad_format}) (sum = {sum_grads:.3f})'
			f'\nSGD solution: ({point_format})'
			f'\nSciPy solution: ({scipy_solution})',
			ax, lines, point, trajectory
		)

		if sum_grads < threshold:
			break

	plt.show()


if __name__ == '__main__':
	assert DIM in (2, 3)

	ax = plt.axes() if DIM == 2 else plt.axes(projection='3d')

	while True:
		lines = generate_lines(NUM_LINES)
		initial_point = np.random.rand(DIM)

		# 1. Solve with SciPy first

		result = minimize(objective_function, initial_point, args=(lines,))
		optimal_solution = result.x

		if np.all((0 <= optimal_solution) & (optimal_solution <= 1)):
			break

	scipy_solution = ', '.join(f'{i:.3f}' for i in optimal_solution)

	# 2. Animate Stochastic Gradient Descent solution

	solve_with_sgd(initial_point, lines)
