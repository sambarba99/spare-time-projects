"""
Genetic Algorithm applied to the Travelling Salesman Problem

Author: Sam Barba
Created 2026-07-16
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['figure.figsize'] = (12, 5)
np.random.seed(1)

NUM_POINTS = 100  # (N - 1)! / 2 = 4.7x10^155 permutations
GENERATIONS = 700
POP_SIZE = 600
ELITISM_RATE = 0.05  # Proportion of the fittest individuals to avoid mutation
TOURNAMENT_SIZE = 10
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

points = np.random.uniform(0, 100, size=(NUM_POINTS, 2))


@dataclass
class Individual:
	chromosome: np.ndarray = field(default_factory=lambda: np.random.permutation(NUM_POINTS))
	fitness: float | None = None

	def copy(self):
		return Individual(self.chromosome.copy(), self.fitness)


def evaluate_population(population):
	chromosomes = np.array([i.chromosome for i in population])

	# Shape: (POP_SIZE, NUM_POINTS, 2)
	routes = points[chromosomes]

	# Difference between consecutive points
	diff = np.diff(routes, axis=1)

	# Distances between points
	distances = np.linalg.norm(diff, axis=2).sum(axis=1)
	loop_back_distance = np.linalg.norm(routes[:, 0] - routes[:, -1], axis=1)

	fitnesses = -(distances + loop_back_distance)  # Negated, as we want to maximise fitness (minimise distance)

	for i, fitness in zip(population, fitnesses):
		i.fitness = fitness


def select(population):
	"""Tournament selection"""

	contenders = np.random.choice(POP_SIZE, size=TOURNAMENT_SIZE, replace=False)
	winner_idx = max(contenders, key=lambda i: population[i].fitness)

	return population[winner_idx]


def crossover(parent1, parent2):
	"""Order crossover"""

	if np.random.random() < CROSSOVER_RATE:
		start, end = sorted(np.random.choice(NUM_POINTS, size=2, replace=False))
		child = Individual(np.full(NUM_POINTS, -1))

		# Copy segment from parent1
		child.chromosome[start:end + 1] = parent1.chromosome[start:end + 1]

		# Fill remaining positions from parent2
		p2_idx = 0
		for i in range(NUM_POINTS):
			if child.chromosome[i] == -1:
				while parent2.chromosome[p2_idx] in child.chromosome:
					p2_idx += 1
				child.chromosome[i] = parent2.chromosome[p2_idx]
				p2_idx += 1

		return child

	return parent1.copy() if np.random.random() < 0.5 else parent2.copy()


def mutate(individual):
	if np.random.random() < MUTATION_RATE:
		choice = np.random.random()
		i, j = sorted(np.random.choice(NUM_POINTS, size=2, replace=False))

		if choice < 0.33:
			# Inversion mutation
			individual.chromosome[i:j + 1] = individual.chromosome[i:j + 1][::-1]
		elif choice < 0.66:
			# Swap mutation
			individual.chromosome[i], individual.chromosome[j] = individual.chromosome[j], individual.chromosome[i]
		else:
			# Scramble mutation
			np.random.shuffle(individual.chromosome[i:j + 1])


if __name__ == '__main__':
	ax_evolution = plt.axes([0.08, 0.1, 0.5, 0.82])
	ax_solution = plt.axes([0.42, 0.12, 0.75, 0.75])
	ax_solution.axis('scaled')

	population = [Individual() for _ in range(POP_SIZE)]
	evaluate_population(population)

	elite_count = int(POP_SIZE * ELITISM_RATE)

	mean_fitnesses, best_fitnesses = [], []

	for _ in range(GENERATIONS):
		population.sort(key=lambda i: i.fitness, reverse=True)

		new_population = []

		# Elitism
		for i in range(elite_count):
			new_population.append(population[i].copy())

		# Produce offspring
		while len(new_population) < POP_SIZE:
			parent1 = select(population)
			parent2 = select(population)

			child = crossover(parent1, parent2)

			mutate(child)

			new_population.append(child)

		population = new_population
		evaluate_population(population)

		mean_fitness = sum(i.fitness for i in population) / POP_SIZE
		best = max(population, key=lambda i: i.fitness)
		mean_fitnesses.append(mean_fitness)
		best_fitnesses.append(best.fitness)

		ax_evolution.clear()
		ax_evolution.plot(mean_fitnesses, label='Mean fitness')
		ax_evolution.plot(best_fitnesses, label='Best fitness')
		ax_evolution.set_xlabel('Generation')
		ax_evolution.set_ylabel('Fitness')
		ax_evolution.set_title('Mean and best fitness per generation')
		ax_evolution.legend(loc='upper left')

		# Make route into a loop
		best_route = points[best.chromosome]
		coords = np.vstack((best_route, best_route[0]))

		ax_solution.clear()
		ax_solution.plot(*coords.T, color='red', linewidth=1, zorder=1)
		ax_solution.scatter(*coords.T, color='black', s=18, zorder=2)
		ax_solution.set_xticks([])
		ax_solution.set_yticks([])
		ax_solution.set_title(f'Best found solution (dist = {-best.fitness:.2f})')

		plt.draw()
		plt.pause(1e-6)

	plt.show()
