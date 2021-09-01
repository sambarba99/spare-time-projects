# Travelling Salesman
# Author: Sam Barba
# Created 16/09/2021

from copy import deepcopy
import matplotlib.pyplot as plt
import random

GENERATIONS = 20
POP_SIZE = 200
MUTATION_RATE = 0.2
ELITISM_RATE = 0.05 # Proportion of fittest individuals to avoid mutation
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 10

# ---------------------------------------------------------------------------------------------------- #
# ---------------------------------------------  CLASSES  -------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

class Path:
	def __init__(self, sequence):
		self.sequence = sequence
		self.fitness = None

	def calcFitness(self):
		self.fitness = sum(self.sequence[i].euclideanDist(self.sequence[i + 1]) for i in range(len(self.sequence) - 1))

	def __repr__(self):
		return "{}  (dist = {})".format(" -> ".join(str(v.label) for v in self.sequence), self.fitness)

class Vertex:
	def __init__(self, label, x, y):
		self.label = label
		self.x = x
		self.y = y

	def euclideanDist(self, other):
		return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def initialisePopulation(vertices):
	population = []

	for i in range(POP_SIZE):
		random.shuffle(vertices)
		population.append(Path(vertices[:]))

	return population

def evaluate(*population):
	for individual in population:
		individual.calcFitness()

# Tournament selection
def selection(population):
	parents = []

	for i in range(POP_SIZE):
		tournamentIndividuals = random.sample(population, TOURNAMENT_SIZE)
		parents.append(findFittest(tournamentIndividuals))

	return parents

def crossover(parents):
	offspring = deepcopy(parents)

	for i in range(POP_SIZE - 1):
		if random.random() > CROSSOVER_RATE: continue

		start = random.randrange(len(parents[i].sequence))
		end = random.randrange(len(parents[i].sequence))
		while start == end:
			end = random.randrange(len(parents[i].sequence))

		start, end = min(start, end), max(start, end)

		offspringSeq = parents[i].sequence[start:end]
		offspringSeq += [v for v in parents[i + 1].sequence if v not in offspringSeq]

		offspring[i] = Path(offspringSeq)
		evaluate(offspring[i])

	return offspring

def mutation(offspring):
	mutants = deepcopy(offspring)
	mutants.sort(key = lambda ind: ind.fitness)

	for i in range(round(ELITISM_RATE * POP_SIZE), POP_SIZE):
		if random.random() > MUTATION_RATE: continue

		random.shuffle(mutants[i].sequence)

	return mutants

def findFittest(population):
	return min(population, key = lambda ind: ind.fitness)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

while True:
	numV = int(input("How many vertices? "))
	print()

	# Generate random vertices

	allCoords = [(x, y) for x in range(100) for y in range(100)]
	coords = random.sample(allCoords, numV)
	vertices = [Vertex(i + 1, *coords[i]) for i in range(numV)]

	print("\n".join("{} ({},{})".format(v.label, v.x, v.y) for v in vertices))

	# Initialise population and perform GA
	population = initialisePopulation(vertices)

	bestPath = population[0]
	meanFitnesses, bestFitnesses = [], []

	for i in range(GENERATIONS):
		evaluate(*population)
		parents = selection(population)
		offspring = crossover(parents)
		mutants = mutation(offspring)
		population = deepcopy(mutants)

		bestPopPath = findFittest(population)
		if bestPopPath.fitness < bestPath.fitness:
			bestPath = bestPopPath

		meanFitness = sum(ind.fitness for ind in population) / POP_SIZE
		meanFitnesses.append(meanFitness)
		bestFitnesses.append(bestPopPath.fitness)

	print("\n" + str(bestPath))

	# Plot evolution graph

	gens = [i + 1 for i in range(GENERATIONS)]
	plt.plot(gens, meanFitnesses, color = "#0080ff", linewidth = 1)
	plt.plot(gens, bestFitnesses, color = "#008000", linewidth = 1)
	plt.annotate("Mean", (gens[:-2], meanFitnesses[:-2]))
	plt.annotate("Best", (gens[:-2], bestFitnesses[:-2]))
	plt.xlabel("Generation")
	plt.ylabel("Fitness")
	plt.show()

	# Plot shortest path

	x = [v.x for v in bestPath.sequence]
	y = [v.y for v in bestPath.sequence]

	plt.scatter(x, y, color = "black", s = 10, zorder = 2)
	for v in bestPath.sequence:
		plt.annotate(v.label, (v.x, v.y))
	plt.plot(x, y, color = "red", linewidth = 1, zorder = 1)
	plt.show()

	choice = input("\nEnter to continue or X to exit: ").upper()
	if len(choice) > 0 and choice[0] == 'X':
		break
	print()
