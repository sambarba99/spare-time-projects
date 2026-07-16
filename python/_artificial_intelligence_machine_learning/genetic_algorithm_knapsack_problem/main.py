"""
Genetic Algorithm applied to the Knapsack Problem

Author: Sam Barba
Created 2021-09-17
"""

from dataclasses import dataclass
import random

import matplotlib.pyplot as plt


random.seed(1)
plt.rcParams['figure.figsize'] = (8, 5)

TOTAL_ITEMS = 100
MAX_ITEM_WEIGHT = 40
MAX_ITEM_VALUE = 10
KNAPSACK_CAPACITY = 50
GENERATIONS = 100
POP_SIZE = 500
ELITISM_RATE = 0.02  # Proportion of the fittest individuals to avoid mutation
CROSSOVER_RATE = 0.75
MUTATION_RATE = 0.05


@dataclass
class Item:
	idx: int
	weight: float
	value: float

	def __repr__(self):
		return f'idx={self.idx}, weight={self.weight:.2f}, value={self.value:.2f}'


all_items = [
	Item(
		i,
		random.uniform(1, MAX_ITEM_WEIGHT),
		random.uniform(1, MAX_ITEM_VALUE)
	)
	for i in range(TOTAL_ITEMS)
]
total_weight = sum(i.weight for i in all_items)
total_value = sum(i.value for i in all_items)
item_prob = KNAPSACK_CAPACITY / total_weight


class Knapsack:
	def __init__(self, chromosome=None):
		if chromosome is None:
			# 1 means item is in knapsack; 0 means it isn't
			chromosome = [1 if random.random() < item_prob else 0 for _ in all_items]

		self.chromosome = chromosome[:]
		self.total_weight = None
		self.total_value = None
		self.fitness = None
		self.calc_fitness()

	def calc_fitness(self):
		self.total_weight = 0
		self.total_value = 0
		for gene, item in zip(self.chromosome, all_items):
			if gene:
				self.total_weight += item.weight
				self.total_value += item.value

		if self.total_weight <= KNAPSACK_CAPACITY:
			self.fitness = self.total_value
		else:
			self.fitness = 0  # Invalid solution

	def copy(self):
		return Knapsack(self.chromosome)

	def __repr__(self):
		return f'weight={self.total_weight:.2f}, value={self.total_value:.2f}'


def select(population):
	"""Roulette wheel selection"""

	fitnesses = [i.fitness for i in population]

	if max(fitnesses) == 0:
		return random.choice(population)

	return random.choices(population, weights=fitnesses)[0]


def crossover(parent1, parent2):
	"""Single-point crossover"""

	if random.random() < CROSSOVER_RATE:
		point = random.randint(1, len(parent1.chromosome) - 1)

		child1_chromosome = parent1.chromosome[:point] + parent2.chromosome[point:]
		child2_chromosome = parent2.chromosome[:point] + parent1.chromosome[point:]

		return Knapsack(child1_chromosome), Knapsack(child2_chromosome)

	return parent1.copy(), parent2.copy()


def mutate(individual):
	"""Single bit-flip mutation"""

	if random.random() < MUTATION_RATE:
		bit = random.randrange(len(individual.chromosome))
		individual.chromosome[bit] ^= 1
		individual.calc_fitness()


if __name__ == '__main__':
	print('\nAll items:\n')
	for item in all_items:
		print(item)

	print(f'\nTotal weight: {total_weight:.2f}')
	print(f'Total value: {total_value:.2f}')
	print(f'Knapsack capacity: {KNAPSACK_CAPACITY}')

	population = [Knapsack() for _ in range(POP_SIZE)]

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

			child1, child2 = crossover(parent1, parent2)

			mutate(child1)
			mutate(child2)

			new_population.append(child1)
			if len(new_population) < POP_SIZE:
				new_population.append(child2)

		population = new_population

		mean_fitness = sum(i.fitness for i in population) / POP_SIZE
		best_fitness = max(i.fitness for i in population)
		mean_fitnesses.append(mean_fitness)
		best_fitnesses.append(best_fitness)
		plt.cla()
		plt.plot(mean_fitnesses, label='Mean fitness')
		plt.plot(best_fitnesses, label='Best fitness')
		plt.xlabel('Generation')
		plt.ylabel('Fitness')
		plt.title('Mean and best fitness per generation')
		plt.legend()
		plt.draw()
		plt.pause(1e-6)

	population.sort(key=lambda i: i.fitness, reverse=True)
	best = population[0]
	print('\nBest knapsack:', best)
	print('Items:')
	for gene, item in zip(best.chromosome, all_items):
		if gene:
			print(item)

	plt.show()
