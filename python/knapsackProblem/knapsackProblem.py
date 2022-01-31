# Knapsack Problem
# Author: Sam Barba
# Created 17/09/2021

from copy import deepcopy
import matplotlib.pyplot as plt
import random

CAP = 30 # Knapsack capacity
NUM_ITEMS = 10

GENERATIONS = 20
POP_SIZE = 200
MUTATION_RATE = 0.2
ELITISM_RATE = 0.02 # Proportion of the fittest individuals to avoid mutation
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 2

# ---------------------------------------------------------------------------------------------------- #
# ---------------------------------------------  CLASSES  -------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

class Knapsack:
	def __init__(self, itemConfig):
		# E.g. [0, 1, 1, 0] means include 2nd and 3rd items
		self.itemConfig = itemConfig
		self.fitness = None

	def calcFitness(self, allItems):
		self.fitness = self.totalValue(allItems) if self.totalWeight(allItems) <= CAP else 0

	def totalValue(self, allItems):
		return sum(allItems[i].value for i in range(NUM_ITEMS) if self.itemConfig[i] == 1)

	def totalWeight(self, allItems):
		return sum(allItems[i].weight for i in range(NUM_ITEMS) if self.itemConfig[i] == 1)

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

def initialisePopulation():
	population = []

	for i in range(POP_SIZE):
		itemConfig = [random.randint(0, 1) for _ in range(NUM_ITEMS)]
		population.append(Knapsack(itemConfig))

	return population

def evaluate(allItems, *population):
	for individual in population:
		individual.calcFitness(allItems)

# Tournament selection
def selection(population):
	parents = []

	for i in range(POP_SIZE):
		tournamentIndividuals = random.sample(population, TOURNAMENT_SIZE)
		parents.append(findFittest(*tournamentIndividuals))

	return parents

# Single-point crossover
def crossover(parents, allItems):
	offspring = deepcopy(parents)

	for i in range(POP_SIZE - 1):
		if random.random() > CROSSOVER_RATE: continue

		p1config = parents[i].itemConfig
		p2config = parents[i + 1].itemConfig

		c = random.randrange(NUM_ITEMS)

		off1config = p1config[:c] + p2config[c:]
		off2config = p2config[:c] + p1config[c:]

		off1 = Knapsack(off1config)
		off2 = Knapsack(off2config)
		evaluate(allItems, off1, off2)

		offspring[i] = findFittest(off1, off2)

	return offspring

# Single bit-flip mutation
def mutation(offspring):
	mutants = deepcopy(offspring)
	mutants.sort(key=lambda ind: ind.fitness, reverse=True)

	for i in range(round(ELITISM_RATE * POP_SIZE), POP_SIZE):
		if random.random() <= MUTATION_RATE:
			randIdx = random.randrange(NUM_ITEMS)
			mutants[i].itemConfig[randIdx] = 1 - mutants[i].itemConfig[randIdx]

	return mutants

def findFittest(*population):
	return max(population, key=lambda ind: ind.fitness)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

# Generate random items

itemValues = [random.randint(50, 500) for _ in range(NUM_ITEMS)]
itemWeights = [random.randint(1, 15) for _ in range(NUM_ITEMS)]
allItems = [Item(i + 1, itemValues[i], itemWeights[i]) for i in range(NUM_ITEMS)]

for item in allItems:
	print(str(item))

print("\nTotal value:", sum(itemValues))
print("Total weight:", sum(itemWeights))
print("Knapsack capacity:", CAP)

# Initialise population and perform GA
population = initialisePopulation()

bestKnapsack = population[0]
meanFitnesses, bestFitnesses = [], []

for _ in range(GENERATIONS):
	evaluate(allItems, *population)
	parents = selection(population)
	offspring = crossover(parents, allItems)
	mutants = mutation(offspring)
	population = deepcopy(mutants)

	bestPopKnapsack = findFittest(*population)
	if bestPopKnapsack.fitness > bestKnapsack.fitness:
		bestKnapsack = bestPopKnapsack

	meanFitness = sum(ind.fitness for ind in population) / POP_SIZE
	meanFitnesses.append(meanFitness)
	bestFitnesses.append(bestPopKnapsack.fitness)

# Display best items

items = [allItems[i] for i in range(NUM_ITEMS) if bestKnapsack.itemConfig[i] == 1]
print("\nBest items:", ", ".join(str(item.index) for item in items))
print("Value:", sum(item.value for item in items))
print("Weight left:", CAP - sum(item.weight for item in items))

# Plot evolution graph

gens = [i + 1 for i in range(GENERATIONS)]
plt.figure(figsize=(8, 6))
plt.plot(gens, meanFitnesses, color="#0080ff", linewidth=1)
plt.plot(gens, bestFitnesses, color="#008000", linewidth=1)
plt.annotate("Mean", (gens[-2], meanFitnesses[-2]))
plt.annotate("Best", (gens[-2], bestFitnesses[-2]))
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()
