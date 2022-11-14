"""
(TensorFlow) MNIST convolutional neural network

Author: Sam Barba
Created 20/10/2021
"""

# Reduce TensorFlow logger spam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import load_model, Model, Sequential
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.random import set_seed

N_CLASSES = 10  # Class for each digit 0-9
INPUT_SHAPE = (28, 28, 1)  # W, H, colour channels
DRAWING_SIZE = 500

plt.rcParams['figure.figsize'] = (10, 5)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def load_data():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Normalise images to 0-1 range and correct shape
	x = np.concatenate([x_train, x_test], axis=0).astype(float) / 255
	x = np.reshape(x, (len(x), *INPUT_SHAPE))

	# One-hot encode y
	y = np.concatenate([y_train, y_test])
	y = np.eye(N_CLASSES)[y]

	# Train:validation:test ratio of 0.7:0.2:0.1
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, stratify=y, random_state=1)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.78, stratify=y_train, random_state=1)

	return x_train, y_train, x_val, y_val, x_test, y_test

def build_model():
	model = Sequential(
		layers=[
			Input(shape=INPUT_SHAPE),
			Conv2D(32, kernel_size=(3, 3), activation='relu'),
			MaxPooling2D(),  # pool_size = (2, 2)
			Conv2D(64, kernel_size=(3, 3), activation='relu'),
			MaxPooling2D(),
			Flatten(),
			Dropout(0.5),
			Dense(N_CLASSES, activation='softmax')
		],
		name='digit_recognition_model'
	)

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.build(input_shape=INPUT_SHAPE)

	return model

def plot_confusion_matrix(actual, predictions, labels, is_training):
	cm = confusion_matrix(actual, predictions)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
	f1 = f1_score(actual, predictions, average='weighted')

	disp.plot(cmap=plt.cm.plasma)
	plt.title(f'{"Training" if is_training else "Test"} confusion matrix\n(F1 score: {f1})')
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	set_seed(1)

	x_train, y_train, x_val, y_val, x_test, y_test = load_data()

	choice = input('\nEnter T to train a new model or L to load existing one\n>>> ').upper()

	match choice:
		case 'T':
			# Plot some training examples

			_, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
			plt.subplots_adjust(hspace=0.01, wspace=0.1)
			for idx, ax in enumerate(axes.flatten()):
				train_idx = np.where(y_train.argmax(axis=1) == idx)[0][0]
				sample = x_train[train_idx].squeeze()
				ax.imshow(sample, cmap='gray')
				ax.set_xticks([])
				ax.set_yticks([])
			plt.suptitle('Training data examples', x=0.51, y=0.92)
			plt.show()

			# Build model

			model = build_model()
			model.summary()
			plot_model(model, show_shapes=True, show_dtype=True, expand_nested=True, show_layer_activations=True)

			# Train model

			print('\n----- TRAINING -----\n')

			# Define early stopping:
			# - min_delta = min. change in monitored quality to qualify as an improvement
			# - patience = no. epochs with no improvement after which training will stop
			# - restore_best_weights = whether to restore model weights from the epoch with
			# 	the best value of the monitored quantity (validation loss in this case)

			early_stopping = EarlyStopping(monitor='val_loss',
				min_delta=0,
				patience=5,
				restore_best_weights=True)

			history = model.fit(x_train, y_train,
				epochs=50,
				validation_data=(x_val, y_val),
				callbacks=[early_stopping],
				verbose=1)

			# Plot loss and accuracy throughout training
			_, (ax_loss, ax_accuracy) = plt.subplots(nrows=2, sharex=True)
			ax_loss.plot(history.history['loss'], label='Training loss')
			ax_loss.plot(history.history['val_loss'], label='Validation loss')
			ax_accuracy.plot(history.history['accuracy'], label='Training accuracy')
			ax_accuracy.plot(history.history['val_accuracy'], label='Validation accuracy')
			ax_loss.set_ylabel('Categorical\ncross-entropy')
			ax_accuracy.set_ylabel('Accuracy')
			ax_accuracy.set_xlabel('Epoch')
			ax_loss.legend()
			ax_accuracy.legend()
			plt.title('Model loss and accuracy during training', y=2.24)
			plt.show()

			model.save('model.h5')
		case 'L':
			model = load_model('model.h5')
		case _:
			return

	# Evaluate model

	print('\n----- EVALUATION -----\n')
	test_loss, test_accuracy = model.evaluate(x_test, y_test)
	print('Test loss:', test_loss)
	print('Test accuracy:', test_accuracy)

	# Plot confusion matrices

	train_pred = model.predict(x_train)
	test_pred = model.predict(x_test)
	plot_confusion_matrix(y_train.argmax(axis=1), train_pred.argmax(axis=1), range(10), True)
	plot_confusion_matrix(y_test.argmax(axis=1), test_pred.argmax(axis=1), range(10), False)

	# User draws a digit to predict

	pg.init()
	pg.display.set_caption('Draw a digit!')
	scene = pg.display.set_mode((DRAWING_SIZE, DRAWING_SIZE))
	user_drawing_coords = []
	drawing = True
	left_btn_down = False

	while drawing:
		for event in pg.event.get():
			match event.type:
				case pg.QUIT:
					drawing = False
					pg.quit()
				case pg.MOUSEBUTTONDOWN:
					if event.button == 1:
						left_btn_down = True
						x, y = event.pos
						user_drawing_coords.append([x, y])
						scene.set_at((x, y), (255, 255, 255))
						pg.display.update()
				case pg.MOUSEMOTION:
					if left_btn_down:
						x, y = event.pos
						user_drawing_coords.append([x, y])
						scene.set_at((x, y), (255, 255, 255))
						pg.display.update()
				case pg.MOUSEBUTTONUP:
					if event.button == 1:
						left_btn_down = False

	user_drawing_coords = np.array(user_drawing_coords) // (DRAWING_SIZE // 27)  # Make coords range from 0-27
	user_drawing_coords = np.unique(user_drawing_coords, axis=0)  # Keep unique pairs only
	drawn_digit_grid = np.zeros((28, 28))
	drawn_digit_grid[user_drawing_coords[:, 1], user_drawing_coords[:, 0]] = 1
	drawn_digit_input = drawn_digit_grid.reshape((1, *INPUT_SHAPE))
	pred_vector = model.predict(drawn_digit_input)

	plt.imshow(drawn_digit_grid, cmap='gray')
	plt.title(f'Drawn digit is {pred_vector.argmax()} ({(100 * pred_vector.max()):.3f}% sure)')
	plt.show()

	# Plot filters and conv layer feature maps for user-drawn digit

	conv_layer_indices = [idx for idx, layer_name in enumerate([layer.name for layer in model.layers]) if 'conv' in layer_name]

	for idx in conv_layer_indices:
		layer = model.layers[idx]
		filters, biases = layer.get_weights()
		n_filters = filters.shape[-1]
		rows = n_filters // 8  # Works because n_filters is 32 for 1st conv layer, 64 for 2nd one
		cols = n_filters // rows

		print(f'\nLayer name: {layer.name} | Filters shape: {filters.shape} | Biases shape: {biases.shape}', end='')

		_, axes = plt.subplots(nrows=rows, ncols=cols)
		for ax_idx, ax in enumerate(axes.flatten()):
			filt = filters[:, :, :, ax_idx]
			ax.set_xticks([])
			ax.set_yticks([])
			ax.imshow(filt[:, :, 0], cmap='gray')  # Plot only 0th (red) channel
		plt.suptitle(f"Filters for '{layer.name}' layer", x=0.512, y=0.94)
		plt.show()

	outputs = [model.layers[i].output for i in conv_layer_indices]  # Conv outputs for user digit
	short_model = Model(inputs=model.inputs, outputs=outputs)
	feature_output = short_model.predict(drawn_digit_input)

	"""
	feature_output = output of each convolutional layer (2 feature maps)
	Shape of 1st: (1, 26, 26, 32) (1: just 1 image; 26x26: WxH; 32: depth)
	Shape of 2nd: (1, 11, 11, 64)

	Calculating their shape:

	1. Conv layers accept a volume of size W1 x H1 x D1.
	   They require 4 hyperparameters:
	      - No. filters K (32 and 64 here)
	      - Filter/kernel size F (both 3 here)
	      - Stride S (default 1)
	      - Amount of zero padding P (default 0)
	   A volume W2 x H2 x D2 is produced, where:
	      W2 = (W1 - F + 2P) / S + 1
	      H2 = (H1 - F + 2P) / S + 1
	      D2 = K

	2. Pooling layers accept a volume of size W1 x H1 x D1.
	   They require 2 hyperparameters:
	      - Filter/kernel size F (both 2 here)
	      - Stride S (defaults to F if not specified, so 2)
	   A volume W2 x H2 x D2 is produced, where:
	      W2 = (W1 - F) / S + 1
	      H2 = (H1 - F) / S + 1
	      D2 = D1

	So shape of 1st conv layer output: W2 = H2 = (28 - 3 + 2(0)) / 1 + 1 = 26
	1st pooling layer output: W2 = H2 = (26 - 2) / 2 + 1 = 13
	2nd conv layer output: W2 = H2 = (13 - 3 + 2(0)) / 1 + 1 = 11
	"""

	for idx, feature_map in enumerate(feature_output, start=1):
		print(f'\nFeature map {idx}/{len(feature_output)} shape: {feature_map.shape}', end='')

		map_depth = feature_map.shape[-1]
		rows = map_depth // 8
		cols = map_depth // rows

		_, axes = plt.subplots(nrows=rows, ncols=cols)
		for ax_idx, ax in enumerate(axes.flatten()):
			ax.set_xticks([])
			ax.set_yticks([])
			ax.imshow(feature_map[0, :, :, ax_idx], cmap='gray')  # Plot feature_map of depth 'ax_idx'
		plt.suptitle(f'Feature map of convolutional layer {idx}/{len(feature_output)}\n(user-drawn digit)', x=0.512, y=0.97)
		plt.show()

if __name__ == '__main__':
	main()