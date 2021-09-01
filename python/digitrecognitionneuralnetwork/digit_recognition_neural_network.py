"""
Demo of a hand-coded digit recodnition neural network

Author: Sam Barba
Created 20/10/2021
"""

# 1 row in data (MNIST dataset) contains 784 pixel values (i.e. 28*28 image) from 0-255, and a class label (0-9)

import matplotlib.pyplot as plt
from neural_network_classifier import NeuralNetwork
import numpy as np
import pygame as pg
from time import perf_counter

DRAWING_SIZE = 500

plt.rcParams['figure.figsize'] = (8, 6)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def extract_data(train_test_ratio=0.8):
	"""Split file data into train/test"""

	data = np.genfromtxt('C:\\Users\\Sam Barba\\Desktop\\Programs\\datasets\\mnist.txt',
		dtype=str, delimiter='\n')
	# Skip header and convert to floats
	data = [row.split() for row in data[1:]]
	data = np.array(data).astype(float)
	np.random.shuffle(data)

	split = int(len(data) * train_test_ratio)

	training_set, test_set = data[:split], data[split:]

	x_train = training_set[:, :-1] / 255
	y_train = training_set[:, -1].astype(int)
	x_test = test_set[:, :-1] / 255
	y_test = test_set[:, -1].astype(int)

	y_train1 = np.zeros((len(y_train), 10))
	y_train1[np.arange(len(y_train)), y_train] = 1
	y_test1 = np.zeros((len(y_test), 10))
	y_test1[np.arange(len(y_test)), y_test] = 1

	y_train, y_test = y_train1, y_test1
	
	return x_train, y_train, x_test, y_test

def confusion_matrix(predictions, actual):
	predictions = np.argmax(predictions, axis=1)
	actual = np.argmax(actual, axis=1)
	n_classes = len(np.unique(actual))
	conf_mat = np.zeros((n_classes, n_classes)).astype(int)

	for a, p in zip(actual, predictions):
		conf_mat[a][p] += 1

	accuracy = np.trace(conf_mat) / conf_mat.sum()
	return conf_mat, accuracy

def plot_confusion_matrices(train_conf_mat, train_acc, test_conf_mat, test_acc):
	# axes[0] = training confusion matrix
	# axes[1] = test confusion matrix
	_, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
	axes[0].matshow(train_conf_mat, cmap=plt.cm.Blues, alpha=0.7)
	axes[1].matshow(test_conf_mat, cmap=plt.cm.Blues, alpha=0.7)
	axes[0].xaxis.set_ticks_position('bottom')
	axes[1].xaxis.set_ticks_position('bottom')
	for (j, i), val in np.ndenumerate(train_conf_mat):
		axes[0].text(x=i, y=j, s=val, ha='center', va='center')
		axes[1].text(x=i, y=j, s=test_conf_mat[j][i], ha='center', va='center')
	axes[0].set_xlabel('Predictions')
	axes[0].set_ylabel('Actual')
	axes[0].set_title(f'Training Confusion Matrix\nAccuracy = {train_acc:.3f}')
	axes[1].set_title(f'Test Confusion Matrix\nAccuracy = {test_acc:.3f}')
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	x_train, y_train, x_test, y_test = extract_data()

	clf = NeuralNetwork()

	choice = input('Enter F to use file parameters or T to train network from scratch\n>>> ').upper()

	if choice == 'F':
		with open('biases_and_weights.txt', 'r') as file:
			params = file.read().split('\n---\n')

		hidden_bias = np.array(params[0].split('\n')).astype(float).reshape(-1, 1)
		hidden_weights = np.array([line.split() for line in params[1].split('\n')]).astype(float)
		output_bias = np.array(params[2].split('\n')).astype(float).reshape(-1, 1)
		output_weights = np.array([line.split() for line in params[3].split('\n')]).astype(float)

		clf.hidden_bias = hidden_bias
		clf.hidden_weights = hidden_weights
		clf.output_bias = output_bias
		clf.output_weights = output_weights
	elif choice == 'T':
		start = perf_counter()
		clf.fit(x_train, y_train)
		interval = perf_counter() - start

		s = ('\n'.join(str(b) for b in clf.hidden_bias.flatten())
			+ '\n---\n'
			+ '\n'.join((' '.join(str(i) for i in w)) for w in clf.hidden_weights)
			+ '\n---\n'
			+ '\n'.join(str(b) for b in clf.output_bias.flatten())
			+ '\n---\n'
			+ '\n'.join((' '.join(str(i) for i in w)) for w in clf.output_weights))
		with open('biases_and_weights.txt', 'w') as file:
			file.write(s)

		print(f'Done in {interval:.3f}s. Saved biases and weights to file.')
	else:
		print('Bad choice')
		return

	# Plot confusion matrices

	train_predictions = [clf.predict(i) for i in x_train]
	test_predictions = [clf.predict(i) for i in x_test]
	train_conf_mat, train_acc = confusion_matrix(train_predictions, y_train)
	test_conf_mat, test_acc = confusion_matrix(test_predictions, y_test)

	plot_confusion_matrices(train_conf_mat, train_acc, test_conf_mat, test_acc)

	# User draws a digit to predict

	pg.init()
	pg.display.set_caption('Draw a digit!')
	scene = pg.display.set_mode((DRAWING_SIZE, DRAWING_SIZE))
	user_coords = []
	drawing = True
	left_button_down = False

	while drawing:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				drawing = False
				pg.quit()
			elif event.type == pg.MOUSEBUTTONDOWN:
				if event.button == 1:
					left_button_down = True
					x, y = event.pos
					user_coords.append([x, y])
					scene.set_at((x, y), (255, 255, 255))
					pg.display.update()
			elif event.type == pg.MOUSEMOTION:
				if left_button_down:
					x, y = event.pos
					user_coords.append([x, y])
					scene.set_at((x, y), (255, 255, 255))
					pg.display.update()
			elif event.type == pg.MOUSEBUTTONUP:
				if event.button == 1:
					left_button_down = False

	user_coords = np.array(user_coords) // (DRAWING_SIZE // 27)  # Make coords range from 0-27
	user_coords = np.unique(user_coords, axis=0)  # Keep unique pairs only
	drawn_digit_display = np.zeros((28, 28))
	drawn_digit_display[user_coords[:, 1], user_coords[:, 0]] = 1
	drawn_digit = drawn_digit_display.reshape(1, 784)[0].astype(int)
	pred_vector = clf.predict(drawn_digit)

	plt.imshow(drawn_digit_display, cmap='gray')
	plt.title(f'Drawn digit is: {np.argmax(pred_vector)}\n({(100 * np.max(pred_vector)):.1f}% sure)')
	plt.show()

	# Plot loss graph

	if choice == 'T':
		plt.plot(clf.loss, color='red')
		plt.xlabel('Training iteration')
		plt.ylabel('Mean loss')
		plt.title('Loss')
		plt.show()

if __name__ == '__main__':
	main()
