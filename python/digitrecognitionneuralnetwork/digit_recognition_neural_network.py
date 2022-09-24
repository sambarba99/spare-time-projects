"""
MNIST neural network demo

Author: Sam Barba
Created 20/10/2021
"""

# Reduce TensorFlow logger spam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import load_model, Sequential
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

N_CLASSES = 10  # 10 classes for 10 digits
INPUT_SHAPE = (28, 28, 1)
DRAWING_SIZE = 500

plt.rcParams['figure.figsize'] = (12, 6)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def load_data():
	(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

	# Scale images to 0-1 range
	x_train = np.array(x_train) / 255
	x_test = np.array(x_test) / 255

	# Ensure shape (28, 28, 1)
	x_train = np.expand_dims(x_train, -1)
	x_test = np.expand_dims(x_test, -1)

	# Convert Y to binary class matrices
	y_train = keras.utils.to_categorical(y_train, N_CLASSES)
	y_test = keras.utils.to_categorical(y_test, N_CLASSES)

	return x_train, y_train, x_test, y_test

def build_model():
	model = Sequential(name='digit_recognition_model')

	model.add(Input(shape=INPUT_SHAPE))
	model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(units=N_CLASSES, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.build(input_shape=INPUT_SHAPE)

	return model

def confusion_matrix(predictions, actual):
	predictions = np.argmax(predictions, axis=1)
	actual = np.argmax(actual, axis=1)
	conf_mat = np.zeros((N_CLASSES, N_CLASSES)).astype(int)

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
	x_train, y_train, x_test, y_test = load_data()

	choice = input('\nEnter T to train a new model or L to load existing one\n>>> ').upper()

	if choice == 'T':
		# Plot some training examples

		for i in range(3):
			sample = np.squeeze(x_train[i])
			sample_class = np.argmax(y_train[i])
			plt.imshow(sample, cmap='gray')
			plt.title(f'Digit: {sample_class}')
			plt.show()

		# Build model

		model = build_model()
		model.summary()
		plot_model(model, show_shapes=True, show_dtype=True, expand_nested=True, show_layer_activations=True)

		# Train model

		print('\n----- TRAINING -----\n')
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5,	restore_best_weights=True)
		history = model.fit(x_train, y_train, epochs=50, validation_split=0.1, callbacks=[early_stopping], verbose=1)

		final_train_loss = history.history['loss'][-1]
		final_val_loss = history.history['val_loss'][-1]
		plt.plot(history.history['loss'], label='Training')
		plt.plot(history.history['val_loss'], label='Validation')
		plt.legend()
		plt.xlabel('Epoch')
		plt.ylabel('Categorical cross-entropy loss')
		plt.title('Model loss during training'
			f'\nFinal training loss: {final_train_loss:.3f}'
			f'\nFinal validation loss: {final_val_loss:.3f}')
		plt.show()

		choice = input('\nSave this model (will override model.h5 if it exists)? (Y/[N])\n>>> ').upper()
		if choice == 'Y':
			model.save('model.h5')
			print('Saved')
	elif choice == 'L':
		model = load_model('model.h5')
	else:
		return

	# Evaluate model

	print('\n----- EVALUATION -----\n')
	test_loss, test_accuracy = model.evaluate(x_test, y_test)
	print('Test loss:', test_loss)
	print('Test accuracy:', test_accuracy)

	# Plot confusion matrices

	train_predictions = model.predict(x_train)
	test_predictions = model.predict(x_test)
	train_conf_mat, train_acc = confusion_matrix(train_predictions, y_train)
	test_conf_mat, test_acc = confusion_matrix(test_predictions, y_test)

	plot_confusion_matrices(train_conf_mat, train_acc, test_conf_mat, test_acc)

	# User draws a digit to predict

	pg.init()
	pg.display.set_caption('Draw a digit!')
	scene = pg.display.set_mode((DRAWING_SIZE, DRAWING_SIZE))
	user_drawing_coords = []
	drawing = True
	left_btn_down = False

	while drawing:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				drawing = False

			elif event.type == pg.MOUSEBUTTONDOWN:
				if event.button == 1:
					left_btn_down = True
					x, y = event.pos
					user_drawing_coords.append([x, y])
					scene.set_at((x, y), (255, 255, 255))
					pg.display.update()

			elif event.type == pg.MOUSEMOTION:
				if left_btn_down:
					x, y = event.pos
					user_drawing_coords.append([x, y])
					scene.set_at((x, y), (255, 255, 255))
					pg.display.update()

			elif event.type == pg.MOUSEBUTTONUP:
				if event.button == 1:
					left_btn_down = False

	user_drawing_coords = np.array(user_drawing_coords) // (DRAWING_SIZE // 27)  # Make coords range from 0-27
	user_drawing_coords = np.unique(user_drawing_coords, axis=0)  # Keep unique pairs only
	drawn_digit_grid = np.zeros((28, 28))
	drawn_digit_grid[user_drawing_coords[:, 1], user_drawing_coords[:, 0]] = 1
	drawn_digit_input = drawn_digit_grid.reshape((1, *drawn_digit_grid.shape, 1))
	pred_vector = model.predict(drawn_digit_input)

	plt.imshow(drawn_digit_grid, cmap='gray')
	plt.title(f'Drawn digit is {np.argmax(pred_vector)} ({(100 * np.max(pred_vector)):.1f}% sure)')
	plt.show()

if __name__ == '__main__':
	main()
