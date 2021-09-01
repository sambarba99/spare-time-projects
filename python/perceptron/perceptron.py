# Perceptron
# Author: Sam Barba
# Created 06/09/2021

LEARNING_RATE = 0.4
INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
TARGET = [0, 1, 1, 1] # Learn logical 'OR' function
weights = [0.2, -0.4]

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def step(weights, inputs, threshold = 0):
	x = -threshold

	for w, i in zip(weights, inputs):
		x += w * i

	return 1 if x > threshold else 0

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

error = True
epoch = 0

print("{:^11}{:^11}{:^11}{:^11}{:^11}{:^11}{:^11}{:^11}".format("Epoch", "x1", "x2", "Target Y", "Actual Y", "Error", "w1", "w2"))
print("-" * 88)

while error:
	error = False

	for i in range(len(INPUTS)):
		s = step(weights, INPUTS[i])
		errorVal = TARGET[i] - s

		if errorVal != 0:
			error = True
			weights = [w + LEARNING_RATE * inp * errorVal for w, inp in zip(weights, INPUTS[i])]

		print("{:^11}{:^11}{:^11}{:^11}{:^11}{:^11}{:^11}{:^11}".format((epoch + 1), INPUTS[i][0], INPUTS[i][1], TARGET[i], s, errorVal, weights[0], weights[1]))

	epoch += 1

print("\nSuccess at epoch", epoch)
print("Final weights for x,y:", weights)
