"""
TensorFlow neural network demo for binary classification, multiclass classification, or regression

Author: Sam Barba
Created 09/10/2022
"""

import os

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from _utils.csv_data_loader import load_csv_classification_data, load_csv_regression_data
from _utils.model_evaluation_plots import plot_confusion_matrix, plot_roc_curve


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce tensorflow log spam
plt.rcParams['figure.figsize'] = (8, 5)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)
tf.random.set_seed(1)

NUM_EPOCHS = 100


if __name__ == '__main__':
	task_choice = input(
		'\nEnter B for binary classification,'
		'\nM for multiclass classification,'
		'\nor R for regression\n>>> '
	).upper()
	print()

	match task_choice:
		case 'B':
			dataset_choice = input(
				'Enter 1 for banknote dataset,'
				'\n2 for breast tumour dataset,'
				'\n3 for mushroom dataset,'
				'\n4 for pulsar dataset,'
				'\nor 5 for Titanic dataset\n>>> '
			)
		case 'M':
			dataset_choice = input(
				'Enter G for glass dataset,'
				'\nI for iris dataset,'
				'\nor W for wine dataset\n>>> '
			).upper()
		case 'R':
			dataset_choice = input(
				'Enter B for Boston housing dataset,'
				'\nC for car value dataset,'
				'\nM for medical insurance dataset,'
				'\nor P for Parkinson\'s dataset\n>>> '
			).upper()
		case _:
			raise ValueError('Bad choice')

	match task_choice + dataset_choice:
		case 'B1': path = 'C:/Users/Sam/Desktop/projects/datasets/banknote_authenticity.csv'
		case 'B2': path = 'C:/Users/Sam/Desktop/projects/datasets/breast_tumour_pathology.csv'
		case 'B3': path = 'C:/Users/Sam/Desktop/projects/datasets/mushroom_edibility_classification.csv'
		case 'B4': path = 'C:/Users/Sam/Desktop/projects/datasets/pulsar_identification.csv'
		case 'B5': path = 'C:/Users/Sam/Desktop/projects/datasets/titanic_survivals.csv'
		case 'MG': path = 'C:/Users/Sam/Desktop/projects/datasets/glass_classification.csv'
		case 'MI': path = 'C:/Users/Sam/Desktop/projects/datasets/iris_classification.csv'
		case 'MW': path = 'C:/Users/Sam/Desktop/projects/datasets/wine_classification.csv'
		case 'RB': path = 'C:/Users/Sam/Desktop/projects/datasets/boston_housing.csv'
		case 'RC': path = 'C:/Users/Sam/Desktop/projects/datasets/car_valuation.csv'
		case 'RM': path = 'C:/Users/Sam/Desktop/projects/datasets/medical_costs.csv'
		case 'RP': path = 'C:/Users/Sam/Desktop/projects/datasets/parkinsons_scale.csv'
		case _:
			raise ValueError('Bad choice')

	labels = None
	if task_choice in 'BM':
		x_train, y_train, x_val, y_val, x_test, y_test, labels, _ = \
			load_csv_classification_data(path, train_size=0.7, val_size=0.2, test_size=0.1, x_transform=StandardScaler(), one_hot_y=True)
	else:
		x_train, y_train, x_val, y_val, x_test, y_test, _ = \
			load_csv_regression_data(path, train_size=0.7, val_size=0.2, test_size=0.1, x_transform=StandardScaler())

	# 1. Build model

	num_features = x_train.shape[1]
	num_targets = 1 if task_choice in 'BR' else len(np.unique(y_train, axis=0))

	match task_choice + dataset_choice:
		case 'B1' | 'B2' | 'B3' | 'B4':  # Banknote, breast tumour, mushroom, or pulsar dataset
			model = Sequential([
				Dense(8, input_shape=(num_features,), activation='relu'),
				Dense(num_targets, input_shape=(num_features,), activation='sigmoid')
			])
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

		case 'B4':  # Titanic dataset
			model = Sequential([
				Dense(8, input_shape=(num_features,), activation='relu'),
				Dropout(0.1),
				Dense(num_targets, input_shape=(num_features,), activation='sigmoid')
			])
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

		case 'MG' | 'MI':  # Glass or iris dataset
			model = Sequential([
				Dense(64, input_shape=(num_features,), activation='relu'),
				Dense(64, input_shape=(num_features,), activation='relu'),
				Dense(num_targets, input_shape=(num_features,), activation='softmax')
			])
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		case 'MW':  # Wine dataset
			model = Sequential([
				Dense(16, input_shape=(num_features,), activation='relu'),
				Dense(num_targets, input_shape=(num_features,), activation='softmax')
			])
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		case 'RB':  # Boston housing dataset
			model = Sequential([
				Dense(256, input_shape=(num_features,), activation='relu'),
				Dropout(0.1),
				Dense(num_targets, input_shape=(num_features,), activation='linear')
			])
			model.compile(loss='mse', metrics='mae')

		case 'RC':  # Car value dataset
			model = Sequential([
				Dense(256, input_shape=(num_features,), activation='relu'),
				Dense(256, input_shape=(num_features,), activation='relu'),
				Dense(num_targets, input_shape=(num_features,), activation='linear')
			])
			model.compile(loss='mse', metrics='mae')

		case _:  # Medical insurance or Parkinson's dataset
			model = Sequential([
				Dense(4096, input_shape=(num_features,), activation='relu'),
				Dense(num_targets, input_shape=(num_features,), activation='linear')
			])
			model.compile(loss='mse', metrics='mae')

	model.build(input_shape=(num_features,))
	model.summary()
	# plot_model(model, show_shapes=True, expand_nested=True, show_layer_activations=True)

	# 2. Train model

	print('\n----- TRAINING -----\n')

	# monitor = val_loss, min_delta = 0
	early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

	history = model.fit(
		x_train, y_train,
		epochs=NUM_EPOCHS,
		validation_data=(x_val, y_val),
		callbacks=[early_stopping],
		verbose=0
	)

	# Plot loss and accuracy/MAE throughout training

	loss = [history.history['loss'], history.history['val_loss']]
	metric = [history.history['accuracy'], history.history['val_accuracy']] \
		if task_choice in 'BM' \
		else [history.history['mae'], history.history['val_mae']]

	_, (ax_loss, ax_metric) = plt.subplots(nrows=2, sharex=True)
	ax_loss.plot(loss[0], label='Training loss')
	ax_loss.plot(loss[1], label='Validation loss')
	ax_metric.plot(metric[0], label='Training accuracy' if task_choice in 'BM' else 'Training MAE')
	ax_metric.plot(metric[1], label='Validation accuracy' if task_choice in 'BM' else 'Validation MAE')
	ax_metric.set_xlabel('Epoch')
	ax_metric.set_ylabel('Accuracy' if task_choice in 'BM' else 'MAE')
	if task_choice == 'B': ax_loss.set_ylabel('Binary\ncross-entropy')
	elif task_choice == 'M': ax_loss.set_ylabel('Categorical\ncross-entropy')
	else: ax_loss.set_ylabel('MSE')
	ax_loss.legend()
	ax_metric.legend()
	plt.suptitle(f'Loss and {"accuracy" if task_choice in "BM" else "MAE"} during training', y=0.95)
	plt.show()

	# 3. Evaluation

	print('----- EVALUATION -----\n')

	if task_choice in 'BM':
		test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
		print('Test loss:', test_loss)
		print('Test accuracy:', test_accuracy)
	else:
		test_mse, test_mae = model.evaluate(x_test, y_test, verbose=0)
		print('Test MSE:', test_mse)
		print('Test MAE:', test_mae)

	# 4. Test model

	if task_choice in 'BM':
		print('\n----- TESTING -----')

		test_pred_probs = model.predict(x_test).squeeze()

		if task_choice == 'B':  # Binary
			test_pred_labels = test_pred_probs.round()

			# ROC curve
			plot_roc_curve(y_test, test_pred_probs)
		else:  # Multiclass
			y_test = y_test.argmax(axis=1)
			test_pred_labels = test_pred_probs.argmax(axis=1)

		# Confusion matrix
		f1 = f1_score(y_test, test_pred_labels, average='binary' if len(labels) == 2 else 'weighted')
		plot_confusion_matrix(y_test, test_pred_labels, labels, f'Test confusion matrix\n(F1 score: {f1:.3f})')

	# To save/load a model:
	# model.save('./model.h5')
	# new_model = keras.models.load_model('./model.h5')
