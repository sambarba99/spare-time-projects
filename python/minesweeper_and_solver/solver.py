"""
Solver class

Author: Sam Barba
Created 28/01/2025
"""

from math import comb

import numpy as np


class Solver:
	def __init__(self, game):
		self.game = game
		self.seen_edge_configs = dict()

	def calculate_mine_probs(self):
		# Reset values that aren't fully deterministic (no need to re-compute)
		# and values for open cells (otherwise it will confuse calculations for closed cells)
		for cell in self.game:
			if cell.mine_prob not in (0, 100) or cell.is_open:
				cell.num_mine_configs = 0
				cell.mine_prob = None

		# Run some basic logic rules
		self.count_edges()
		self.rule1()
		r2 = r3 = r4 = True
		while r2 or r3 or r4:
			# Repeat these rules until no further deductions can be made
			r2 = self.rule2()
			r3 = self.rule3()
			r4 = self.rule4()

		# Find remaining edge cells with unknown probs (i.e. not computed by rules 1-4)
		edge_cell_coords = [
			(cell.y, cell.x) for cell in self.game
			if cell.is_edge and cell.mine_prob is None
		]

		# Generate all possible mine/non-mine configurations of edge_cell_coords
		# (e.g. (0,5)=mine, (1,5)=not_mine etc.) and use this to calculate final probs
		if edge_cell_coords:
			if (coords_tpl := tuple(edge_cell_coords)) in self.seen_edge_configs:
				# Seen these coords already, so read possible configs from seen_edge_configs
				possible_configs = self.seen_edge_configs[coords_tpl]
			else:
				possible_configs = self.generate_edge_configurations(edge_cell_coords)
				self.seen_edge_configs[coords_tpl] = possible_configs
			self.calculate_probs_from_configurations(possible_configs, edge_cell_coords)

	def count_edges(self):
		"""Count how many edges are around each open cell"""

		for cell in self.game:
			count = 0
			if cell.is_open:
				cell.is_edge = False
				for neighbour in cell.neighbours:
					if not neighbour.is_open:
						neighbour.is_edge = True
						count += 1
			cell.num_edges = count

	def rule1(self):
		"""Label neighbour probs around isolated open cells"""

		for cell in self.game:
			if cell.num_edges >= 3:
				count = 0
				for neighbour in cell.neighbours:
					nn = neighbour.neighbours
					num_open_nns = sum(nni.is_open for nni in nn)
					if neighbour.is_edge and num_open_nns == 1:
						count += 1
				if count == cell.num_edges:
					mine_prob = round(cell.num_surrounding_mines / cell.num_edges * 100)
					for neighbour in cell.neighbours:
						neighbour.mine_prob = mine_prob

	def rule2(self):
		"""
		Label cells that must be a mine:
		if the mine number of an open cell = number of edge cells around it - prob 0 neighbours,
		then for each neighbour with an unknown mine prob, this prob must be 100%
		"""

		ret = False
		for cell in self.game:
			num_0_prob_neighbours = self.probability_0_count(cell)
			if cell.num_edges > 0 and cell.num_surrounding_mines == cell.num_edges - num_0_prob_neighbours:
				for neighbour in cell.neighbours:
					if neighbour.is_edge and neighbour.mine_prob is None:
						neighbour.mine_prob = 100
						ret = True

		return ret

	def rule3(self):
		"""
		Label cells that must not be a mine:
		if the mine number of an open cell = number of prob 100 neighbours,
		then for each neighbour with an unknown mine prob, this prob must be 0%
		"""

		ret = False
		for cell in self.game:
			num_100_prob_neighbours = self.probability_100_count(cell)
			if cell.num_edges > 0 and cell.num_surrounding_mines == num_100_prob_neighbours:
				for neighbour in cell.neighbours:
					if neighbour.is_edge and neighbour.mine_prob is None:
						neighbour.mine_prob = 0
						ret = True

		return ret

	def rule4(self):
		"""
		Solve for unknown edges by treating them as a system of linear equations,
		then using augmented matrix maths to deduce mines/non-mines

		Source: https://massaioli.wordpress.com/2013/01/12/solving-minesweeper-with-matricies/
		"""

		def rref(matrix):
			"""Convert an augmented matrix to reduced row echelon form"""

			ret = matrix.copy()
			rows, cols = matrix.shape
			curr_row = 0

			for col in range(cols - 1):  # Avoid the augmented column
				# Find the row with the largest element in the current column
				max_row_idx = np.abs(ret[curr_row:rows, col]).argmax() + curr_row
				if ret[max_row_idx, col] == 0:
					continue  # Skip this column, no pivot can be made

				# Swap the current row with the row containing the largest element
				ret[[curr_row, max_row_idx]] = ret[[max_row_idx, curr_row]]
				# Scale the pivot row to make the pivot equal to 1
				ret[curr_row] /= ret[curr_row, col]

				# Eliminate all other entries in the current column
				for i in range(rows):
					if i != curr_row:
						ret[i] -= ret[i, col] * ret[curr_row]

				curr_row += 1
				if curr_row >= rows:
					break

			return ret

		ret = False
		numbered_cells = []
		numbered_cell_neighbours = []

		for cell in self.game:
			if cell.is_open and cell.num_edges > 0:
				edge_neighbours = [
					neighbour for neighbour in cell.neighbours
					if neighbour.is_edge and neighbour.mine_prob is None
				]
				if edge_neighbours:
					numbered_cells.append(cell)
					numbered_cell_neighbours.extend((n.y, n.x) for n in edge_neighbours)

		if not numbered_cells:
			return False

		numbered_cell_neighbours = sorted(set(numbered_cell_neighbours))
		# print(f'\n\n\nUnknown neighbours of number cells:\n{numbered_cell_neighbours}')

		matrix = np.zeros((len(numbered_cells), len(numbered_cell_neighbours)))
		mine_nums = []

		for row_idx, cell in enumerate(numbered_cells):
			for neighbour in cell.neighbours:
				if (neighbour.y, neighbour.x) in numbered_cell_neighbours:
					col_idx = numbered_cell_neighbours.index((neighbour.y, neighbour.x))
					matrix[row_idx][col_idx] = 1
			mine_nums.append(cell.num_surrounding_mines - self.probability_100_count(cell))

		mine_nums = np.array([mine_nums]).T
		augmented_matrix = np.hstack((matrix, mine_nums))
		# print(f'\nAugmented matrix before RREF conversion:\n{augmented_matrix}')

		reduced_matrix = rref(augmented_matrix)
		reduced_matrix = reduced_matrix[reduced_matrix.any(axis=1)]  # Keep only nonzero rows
		# print(f'\nConverted to RREF:\n{reduced_matrix}')

		for row in reduced_matrix:
			coeffs = row[:-1]
			augmented_val = row[-1]
			lower_bound = coeffs[coeffs < 0].sum()
			upper_bound = coeffs[coeffs > 0].sum()
			if augmented_val not in (lower_bound, upper_bound):
				continue
			for neighbour, c in zip(numbered_cell_neighbours, coeffs):
				y, x = neighbour
				if self.game.grid[y][x].mine_prob is None:
					if (augmented_val == lower_bound and c < 0) or (augmented_val == upper_bound and c > 0):
						# print(f'({y}, {x}) = mine')
						self.game.grid[y][x].mine_prob = 100
						ret = True
					elif (augmented_val == lower_bound and c > 0) or (augmented_val == upper_bound and c < 0):
						# print(f'({y}, {x}) = not mine')
						self.game.grid[y][x].mine_prob = 0
						ret = True

		return ret

	def generate_edge_configurations(self, edge_cell_coords):
		"""Given a list of edge cells, generate all possible mine/non-mine configs of these cells"""

		orig_config = [None] * len(edge_cell_coords)  # E.g. 0,0,1,0 means only the cell at index 2 is a mine
		stack = [(orig_config, 0)]  # Configuration, start index
		possible_configs = []

		while stack:
			config, idx = stack.pop()
			y, x = edge_cell_coords[idx]

			if self.can_be_mine(edge_cell_coords, config, y, x):
				pattern_yes = config[:]
				pattern_yes[idx] = 1
				if idx < len(edge_cell_coords) - 1:
					stack.append((pattern_yes, idx + 1))
				else:
					possible_configs.append(pattern_yes)
			if self.can_be_non_mine(edge_cell_coords, config, y, x):
				pattern_no = config[:]
				pattern_no[idx] = 0
				if idx < len(edge_cell_coords) - 1:
					stack.append((pattern_no, idx + 1))
				else:
					possible_configs.append(pattern_no)

		return possible_configs

	def can_be_mine(self, edge_cell_coords, config, y, x):
		"""Determine if a cell can be a mine by looking at open neighbours"""

		for neighbour in self.game.grid[y][x].neighbours:
			if neighbour.is_open:
				num_potential_mines = sum(
					self.game.grid[iy][ix] in neighbour.neighbours
					for (iy, ix), is_mine in zip(edge_cell_coords, config)
					if is_mine == 1
				)
				num_100_prob_neighbours = self.probability_100_count(neighbour)
				total_mines = num_potential_mines + num_100_prob_neighbours
				if total_mines >= neighbour.num_surrounding_mines:
					return False
		return True

	def can_be_non_mine(self, edge_cell_coords, config, y, x):
		"""Determine if a cell can be a non-mine by looking at open neighbours"""

		for neighbour in self.game.grid[y][x].neighbours:
			if neighbour.is_open:
				num_potential_non_mines = sum(
					self.game.grid[iy][ix] in neighbour.neighbours
						for (iy, ix), is_mine in zip(edge_cell_coords, config)
						if is_mine == 0
				)
				num_0_prob_neighbours = self.probability_0_count(neighbour)
				total_non_mines = neighbour.num_edges - num_potential_non_mines - num_0_prob_neighbours
				if neighbour.num_surrounding_mines >= total_non_mines:
					return False
		return True

	def calculate_probs_from_configurations(self, possible_configs, edge_cell_coords):
		"""Calculate probabilities from generated edge configurations"""

		total_non_edge = sum(not (cell.is_open or cell.is_edge) for cell in self.game)
		total_100_prob = sum(cell.mine_prob == 100 for cell in self.game)
		total_possible_combos = 0

		# For each edge cell, count the no. possible configurations in which it's a mine

		for edge_config in possible_configs:
			mines_placed = sum(edge_config)
			remaining_mines = self.game.num_mines - mines_placed - total_100_prob
			if 0 <= remaining_mines <= total_non_edge:
				# How many different ways can we place the remaining mines in the undiscovered region (non-edges)?
				non_edge_combos = comb(total_non_edge, remaining_mines)
				total_possible_combos += non_edge_combos
				for is_mine, (y, x) in zip(edge_config, edge_cell_coords):
					if is_mine:
						self.game.grid[y][x].num_mine_configs += non_edge_combos

		# For each edge cell, its mine probability is:
		# (no. configurations where cell is a mine) / (total possible mine combinations for undiscovered region)

		for cell in self.game:
			if cell.is_edge and cell.mine_prob is None:
				cell.mine_prob = round(cell.num_mine_configs / total_possible_combos * 100)

	def probability_0_count(self, cell):
		"""Count how many surrounding cells have a mine prob of 0%"""

		count = sum(neighbour.mine_prob == 0 for neighbour in cell.neighbours)
		return count

	def probability_100_count(self, cell):
		"""Count how many surrounding cells have a mine prob of 100%"""

		count = sum(neighbour.mine_prob == 100 for neighbour in cell.neighbours)
		return count
