"""
Demo of a Genetic Algorithm applied to the Knapsack Problem

Author: Sam Barba
Created 17/09/2021
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from item import Item
from knapsack import Knapsack

KNAPSACK_CAPACITY = 100
N_ITEMS = 100
GENERATIONS = 50
POP_SIZE = 100
MUTATION_RATE = 0.4
ELITISM_RATE = 0.02  # Proportion of the fittest individuals to avoid mutation
CROSSOVER_RATE = 0.8

all_items = None

plt.rcParams['figure.figsize'] = (7, 5)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def initialise_population():
	"""Initialise population of individuals with random items"""

	population = []

	for _ in range(POP_SIZE):
		individual = Knapsack([False] * N_ITEMS)
		indices = np.random.permutation(N_ITEMS)
		items_shuffle = [(i, all_items[i]) for i in indices]

		for idx, item in items_shuffle:
			if individual.total_weight(all_items, N_ITEMS) + item.weight <= KNAPSACK_CAPACITY:
				individual.item_config[idx] = True

		population.append(individual)

	return population

def selection(population):
	"""Roulette wheel selection"""

	total_fitness = sum([p.fitness for p in population])
	probs = [p.fitness / total_fitness for p in population]
	parents = list(np.random.choice(population, size=POP_SIZE, p=probs))

	return parents

def crossover(parents):
	"""Single-point crossover"""

	offspring = deepcopy(parents)

	for i in range(POP_SIZE - 1):
		if np.random.random() > CROSSOVER_RATE: continue

		p1_config = parents[i].item_config
		p2_config = parents[i + 1].item_config

		c = np.random.choice(N_ITEMS)

		off1_config = p1_config[:c] + p2_config[c:]
		off2_config = p2_config[:c] + p1_config[c:]

		off1 = Knapsack(off1_config)
		off2 = Knapsack(off2_config)
		evaluate(off1, off2)

		offspring[i] = find_fittest(off1, off2)

	return offspring

def mutation(offspring):
	"""Single bit-flip mutation"""

	mutants = deepcopy(offspring)
	mutants.sort(key=lambda ind: ind.fitness, reverse=True)

	for i in range(int(ELITISM_RATE * POP_SIZE), POP_SIZE):
		if np.random.random() > MUTATION_RATE: continue

		r = np.random.choice(N_ITEMS)
		mutants[i].item_config[r] = not mutants[i].item_config[r]

	return mutants

def evaluate(*population):
	if isinstance(population[0], list): population = population[0]

	for individual in population:
		individual.calc_fitness(all_items, N_ITEMS, KNAPSACK_CAPACITY)

def find_fittest(*population):
	if isinstance(population[0], list): population = population[0]

	return max(population, key=lambda ind: ind.fitness)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	global all_items

	np.random.seed(1)

	# Generate random items
	item_weights = np.random.uniform(1, KNAPSACK_CAPACITY, size=N_ITEMS)
	item_values = np.random.uniform(1, 100, size=N_ITEMS)
	all_items = [Item(i + 1, item_weights[i], item_values[i]) for i in range(N_ITEMS)]

	for item in all_items:
		print(item)

	print('\nTotal value:', sum(item_values))
	print('Total weight:', sum(item_weights))
	print('Knapsack capacity:', KNAPSACK_CAPACITY)

	population = initialise_population()

	best_knapsack = population[0]
	mean_fitnesses, best_fitnesses = [], []

	for _ in range(GENERATIONS):
		evaluate(population)
		parents = selection(population)
		offspring = crossover(parents)
		evaluate(offspring)
		mutants = mutation(offspring)
		population = deepcopy(mutants)
		evaluate(population)

		best_pop_knapsack = find_fittest(population)
		if best_pop_knapsack.fitness > best_knapsack.fitness:
			best_knapsack = best_pop_knapsack

		mean_fitness = sum(ind.fitness for ind in population) / POP_SIZE
		mean_fitnesses.append(mean_fitness)
		best_fitnesses.append(best_pop_knapsack.fitness)

		# Plot evolution graph
		plt.cla()
		plt.plot(mean_fitnesses, label='Mean fitness')
		plt.plot(best_fitnesses, label='Best fitness')
		plt.xlabel('Generation')
		plt.ylabel('Fitness')
		plt.title('Mean and best fitness per generation')
		plt.legend()

		plt.show(block=False)
		plt.pause(0.1)

	# Display best items

	items = [all_items[i] for i in range(N_ITEMS) if best_knapsack.item_config[i]]
	print('\nBest items:', ', '.join(str(item.index) for item in items))
	print('Value:', sum(item.value for item in items))
	print('Weight left:', KNAPSACK_CAPACITY - sum(item.weight for item in items))

	plt.show()

if __name__ == '__main__':
	main()
