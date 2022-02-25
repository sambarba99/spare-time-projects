# Knapsack Problem
# Author: Sam Barba
# Created 17/09/2021

from copy import deepcopy
import matplotlib.pyplot as plt
import random

KNAPSACK_CAPACITY = 50
NUM_ITEMS = 100

GENERATIONS = 50
POP_SIZE = 100
MUTATION_RATE = 0.2
ELITISM_RATE = 0.2  # Proportion of the fittest individuals to avoid mutation
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 2

# ---------------------------------------------------------------------------------------------------- #
# ---------------------------------------------  CLASSES  -------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

class Knapsack:
	def __init__(self, item_config):
		# E.g. [False, True, True, False] means include 2nd and 3rd items
		self.item_config = item_config
		self.fitness = None

	def calc_fitness(self):
		self.fitness = self.total_value() if self.total_weight() <= KNAPSACK_CAPACITY else 0

	def total_value(self):
		global all_items
		return sum(all_items[i].value for i in range(NUM_ITEMS) if self.item_config[i])

	def total_weight(self):
		global all_items
		return sum(all_items[i].weight for i in range(NUM_ITEMS) if self.item_config[i])

class Item:
	def __init__(self, index, value, weight):
		self.index = index
		self.value = value
		self.weight = weight

	def __repr__(self):
		return f"Item {self.index}:  value: {self.value}  weight: {self.weight}"

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def initialise_population():
	global all_items
	population = []

	for _ in range(POP_SIZE):
		individual = Knapsack([False] * NUM_ITEMS)
		all_items_copy = all_items[:]
		all_items_copy = list(enumerate(all_items_copy))
		random.shuffle(all_items_copy)

		for idx, item in all_items_copy:
			if individual.total_weight() + item.weight <= KNAPSACK_CAPACITY:
				individual.item_config[idx] = True

		population.append(individual)

	return population

def evaluate(*population):
	for individual in population:
		individual.calc_fitness()

# Tournament selection
def selection(population):
	parents = []

	for i in range(POP_SIZE):
		tournament_individuals = random.sample(population, TOURNAMENT_SIZE)
		parents.append(find_fittest(*tournament_individuals))

	return parents

# Single-point crossover
def crossover(parents):
	offspring = deepcopy(parents)

	for i in range(POP_SIZE - 1):
		if random.random() > CROSSOVER_RATE: continue

		p1_config = parents[i].item_config
		p2_config = parents[i + 1].item_config

		c = random.randrange(NUM_ITEMS)

		off1_config = p1_config[:c] + p2_config[c:]
		off2_config = p2_config[:c] + p1_config[c:]

		off1 = Knapsack(off1_config)
		off2 = Knapsack(off2_config)
		evaluate(off1, off2)

		offspring[i] = find_fittest(off1, off2)

	return offspring

# Single bit-flip mutation
def mutation(offspring):
	mutants = deepcopy(offspring)
	mutants.sort(key=lambda ind: ind.fitness, reverse=True)

	for i in range(int(ELITISM_RATE * POP_SIZE), POP_SIZE):
		if random.random() <= MUTATION_RATE:
			rand_idx = random.randrange(NUM_ITEMS)
			mutants[i].item_config[rand_idx] = not mutants[i].item_config[rand_idx]

	return mutants

def find_fittest(*population):
	return max(population, key=lambda ind: ind.fitness)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Generate random items

item_values = [random.randint(50, 500) for _ in range(NUM_ITEMS)]
item_weights = [random.randint(1, 15) for _ in range(NUM_ITEMS)]
all_items = [Item(i + 1, item_values[i], item_weights[i]) for i in range(NUM_ITEMS)]

for item in all_items:
	print(str(item))

print("\nTotal value:", sum(item_values))
print("Total weight:", sum(item_weights))
print("Knapsack capacity:", KNAPSACK_CAPACITY)

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

# Display best items

items = [all_items[i] for i in range(NUM_ITEMS) if best_knapsack.item_config[i]]
print("\nBest items:", ", ".join(str(item.index) for item in items))
print("Value:", sum(item.value for item in items))
print("Weight left:", KNAPSACK_CAPACITY - sum(item.weight for item in items))

# Plot evolution graph

plt.figure(figsize=(8, 6))
plt.plot(mean_fitnesses, color="#0080ff", lw=1, label="Mean fitness")
plt.plot(best_fitnesses, color="#008000", lw=1, label="Best fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Mean fitness and best fitness of each generation")
plt.legend()
plt.show()
