"""
Demo of a Genetic Algorithm applied to the Knapsack Problem

Author: Sam Barba
Created 17/09/2021
"""

from copy import deepcopy
from item import Item
from knapsack import Knapsack
import matplotlib.pyplot as plt
import random

KNAPSACK_CAPACITY = 50
N_ITEMS = 100
GENERATIONS = 50
POP_SIZE = 100
MUTATION_RATE = 0.01
ELITISM_RATE = 0.01  # Proportion of the fittest individuals to avoid mutation
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 2

all_items = None

plt.rcParams['figure.figsize'] = (7, 5)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def initialise_population():
	population = []

	for _ in range(POP_SIZE):
		individual = Knapsack([False] * N_ITEMS)
		all_items_copy = all_items[:]
		all_items_copy = list(enumerate(all_items_copy))
		random.shuffle(all_items_copy)

		for idx, item in all_items_copy:
			if individual.total_weight(all_items, N_ITEMS) + item.weight <= KNAPSACK_CAPACITY:
				individual.item_config[idx] = True

		population.append(individual)

	return population

def evaluate(*population):
	for individual in population:
		individual.calc_fitness(all_items, N_ITEMS, KNAPSACK_CAPACITY)

def selection(population):
	"""Tournament selection"""
	parents = []

	for i in range(POP_SIZE):
		tournament_individuals = random.sample(population, TOURNAMENT_SIZE)
		parents.append(find_fittest(*tournament_individuals))

	return parents

def crossover(parents):
	"""Single-point crossover"""
	offspring = deepcopy(parents)

	for i in range(POP_SIZE - 1):
		if random.random() > CROSSOVER_RATE: continue

		p1_config = parents[i].item_config
		p2_config = parents[i + 1].item_config

		c = random.randrange(N_ITEMS)

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
		if random.random() <= MUTATION_RATE:
			rand_idx = random.randrange(N_ITEMS)
			mutants[i].item_config[rand_idx] = not mutants[i].item_config[rand_idx]

	return mutants

def find_fittest(*population):
	return max(population, key=lambda ind: ind.fitness)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	global all_items

	# Generate random items
	item_values = [random.randint(10, 500) for _ in range(N_ITEMS)]
	item_weights = [random.randint(1, KNAPSACK_CAPACITY // 2) for _ in range(N_ITEMS)]
	all_items = [Item(i + 1, item_values[i], item_weights[i]) for i in range(N_ITEMS)]

	for item in all_items:
		print(item)

	print('\nTotal value:', sum(item_values))
	print('Total weight:', sum(item_weights))
	print('Knapsack capacity:', KNAPSACK_CAPACITY)

	# Initialise population and perform GA
	population = initialise_population()

	best_knapsack = population[0]
	mean_fitnesses, best_fitnesses = [], []

	for _ in range(GENERATIONS):
		evaluate(*population)
		parents = selection(population)
		offspring = crossover(parents)
		mutants = mutation(offspring)
		population = deepcopy(mutants)

		best_pop_knapsack = find_fittest(*population)
		if best_pop_knapsack.fitness > best_knapsack.fitness:
			best_knapsack = best_pop_knapsack

		mean_fitness = sum(ind.fitness for ind in population) / POP_SIZE
		mean_fitnesses.append(mean_fitness)
		best_fitnesses.append(best_pop_knapsack.fitness)

		# Plot evolution graph
		plt.cla()
		plt.plot(mean_fitnesses, color='#ff8000', label='Mean fitness')
		plt.plot(best_fitnesses, color='#008000', label='Best fitness')
		plt.xlabel('Generation')
		plt.ylabel('Fitness')
		plt.title('Mean and best fitnesses of each generation')
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
