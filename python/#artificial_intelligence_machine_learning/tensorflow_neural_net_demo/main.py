"""
TensorFlow neural network demo for binary classification, multiclass classification, or regression

Author: Sam Barba
Created 09/10/2022
"""

import os

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import vis_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from neural_net_plotter import plot_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce tensorflow log spam
plt.rcParams['figure.figsize'] = (8, 5)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)
tf.random.set_seed(1)

N_EPOCHS = 100


def load_classification_data(path):
	df = pd.read_csv(path)
	print(f'\nRaw data:\n\n{df}')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]
	x_to_encode = x.select_dtypes(exclude=np.number).columns
	labels = sorted(y.unique())

	for col in x_to_encode:
		n_unique = x[col].nunique()
		if n_unique == 1:
			# No information from this feature
			x = x.drop(col, axis=1)
		elif n_unique == 2:
			# Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True).astype(int)
		else:
			# Multivariate feature
			one_hot = pd.get_dummies(x[col], prefix=col).astype(int)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)

	if len(labels) > 2:
		one_hot = pd.get_dummies(y, prefix='class').astype(int)
		y = pd.concat([y, one_hot], axis=1)
		y = y.drop(y.columns[0], axis=1)
	else:  # Binary class
		y = pd.get_dummies(y, prefix='class', drop_first=True).astype(int)

	print(f'\nPreprocessed data:\n\n{pd.concat([x, y], axis=1)}\n')

	# Standardise x
	x, y = x.to_numpy(), y.to_numpy().squeeze()
	# Train:validation:test ratio of 0.7:0.2:0.1
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, stratify=y, random_state=1)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.78, stratify=y_train, random_state=1)
	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)
	x_val = scaler.transform(x_val)

	return x_train, y_train, x_val, y_val, x_test, y_test, labels


def load_regression_data(path):
	df = pd.read_csv(path)
	print(f'\nRaw data:\n\n{df}')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]
	x_to_encode = x.select_dtypes(exclude=np.number).columns

	for col in x_to_encode:
		n_unique = x[col].nunique()
		if n_unique == 1:
			# No information from this feature
			x = x.drop(col, axis=1)
		elif n_unique == 2:
			# Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True).astype(int)
		else:
			# Multivariate feature
			one_hot = pd.get_dummies(x[col], prefix=col).astype(int)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)

	print(f'\nPreprocessed data:\n\n{pd.concat([x, y], axis=1)}\n')

	# Standardise x
	x, y = x.to_numpy(), y.to_numpy()
	# Train:validation:test ratio of 0.7:0.2:0.1
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.78, random_state=1)
	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_test = scaler.transform(x_test)
	x_val = scaler.transform(x_val)

	return x_train, y_train, x_val, y_val, x_test, y_test


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
		case 'B1': path = 'C:/Users/Sam/Desktop/Projects/datasets/banknote_authentication.csv'
		case 'B2': path = 'C:/Users/Sam/Desktop/Projects/datasets/breast_tumour_pathology.csv'
		case 'B3': path = 'C:/Users/Sam/Desktop/Projects/datasets/mushroom_edibility_classification.csv'
		case 'B4': path = 'C:/Users/Sam/Desktop/Projects/datasets/pulsar_identification.csv'
		case 'B5': path = 'C:/Users/Sam/Desktop/Projects/datasets/titanic_survivals.csv'
		case 'MG': path = 'C:/Users/Sam/Desktop/Projects/datasets/glass_classification.csv'
		case 'MI': path = 'C:/Users/Sam/Desktop/Projects/datasets/iris_classification.csv'
		case 'MW': path = 'C:/Users/Sam/Desktop/Projects/datasets/wine_classification.csv'
		case 'RB': path = 'C:/Users/Sam/Desktop/Projects/datasets/boston_housing.csv'
		case 'RC': path = 'C:/Users/Sam/Desktop/Projects/datasets/car_valuation.csv'
		case 'RM': path = 'C:/Users/Sam/Desktop/Projects/datasets/medical_costs.csv'
		case 'RP': path = 'C:/Users/Sam/Desktop/Projects/datasets/parkinsons_scale.csv'
		case _:
			raise ValueError('Bad choice')

	labels = None
	if task_choice in 'BM':
		x_train, y_train, x_val, y_val, x_test, y_test, labels = load_classification_data(path)
	else:
		x_train, y_train, x_val, y_val, x_test, y_test = load_regression_data(path)

	# 1. Build model

	n_features = x_train.shape[1]
	n_targets = 1 if task_choice in 'BR' else len(np.unique(y_train, axis=0))

	match task_choice + dataset_choice:
		case 'B1' | 'B2' | 'B3' | 'B4':  # Banknote, breast tumour, mushroom, or pulsar dataset
			model = Sequential([
				Dense(8, input_shape=(n_features,), activation='relu'),
				Dense(n_targets, input_shape=(n_features,), activation='sigmoid')
			])
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

		case 'B4':  # Titanic dataset
			model = Sequential([
				Dense(8, input_shape=(n_features,), activation='relu'),
				Dropout(0.1),
				Dense(n_targets, input_shape=(n_features,), activation='sigmoid')
			])
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

		case 'MG' | 'MI':  # Glass or iris dataset
			model = Sequential([
				Dense(64, input_shape=(n_features,), activation='relu'),
				Dense(64, input_shape=(n_features,), activation='relu'),
				Dense(n_targets, input_shape=(n_features,), activation='softmax')
			])
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		case 'MW':  # Wine dataset
			model = Sequential([
				Dense(16, input_shape=(n_features,), activation='relu'),
				Dense(n_targets, input_shape=(n_features,), activation='softmax')
			])
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		case 'RB':  # Boston housing dataset
			model = Sequential([
				Dense(256, input_shape=(n_features,), activation='relu'),
				Dropout(0.1),
				Dense(n_targets, input_shape=(n_features,), activation='linear')
			])
			model.compile(loss='mse', metrics='mae')

		case 'RC':  # Car value dataset
			model = Sequential([
				Dense(256, input_shape=(n_features,), activation='relu'),
				Dense(256, input_shape=(n_features,), activation='relu'),
				Dense(n_targets, input_shape=(n_features,), activation='linear')
			])
			model.compile(loss='mse', metrics='mae')

		case _:  # Medical insurance or Parkinson's dataset
			model = Sequential([
				Dense(4096, input_shape=(n_features,), activation='relu'),
				Dense(n_targets, input_shape=(n_features,), activation='linear')
			])
			model.compile(loss='mse', metrics='mae')

	model.build(input_shape=(n_features,))
	model.summary()
	# plot_model(model)
	# vis_utils.plot_model(model, show_shapes=True, expand_nested=True, show_layer_activations=True)

	# 2. Training

	print('\n----- TRAINING -----\n')

	# monitor = val_loss, min_delta = 0
	early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

	history = model.fit(
		x_train, y_train,
		epochs=N_EPOCHS,
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

	# 4. Testing

	if task_choice in 'BM':
		print('\n----- TESTING -----')

		test_pred_probs = model.predict(x_test).squeeze()

		if task_choice == 'B':  # Binary
			test_pred_labels = test_pred_probs.round()

			# ROC curve

			fpr, tpr, _ = roc_curve(y_test, test_pred_probs)
			plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
			plt.plot(fpr, tpr)
			plt.axis('scaled')
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('ROC curve')
			plt.show()
		else:  # Multiclass
			y_test = y_test.argmax(axis=1)
			test_pred_labels = test_pred_probs.argmax(axis=1)

		# Confusion matrix

		cm = confusion_matrix(y_test, test_pred_labels)
		disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
		f1 = f1_score(y_test, test_pred_labels, average='binary' if len(labels) == 2 else 'weighted')

		disp.plot(cmap='plasma')
		plt.title(f'Test confusion matrix\n(F1 score: {f1})')
		plt.show()

	# To save/load a model:
	# model.save('model.h5')
	# new_model = keras.models.load_model('model.h5')
