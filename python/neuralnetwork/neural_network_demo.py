"""
General neural network demo for binary classification, multiclass classification, or regression

Author: Sam Barba
Created 09/10/2022
"""

# Reduce TensorFlow logger spam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Dense, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.random import set_seed

from neural_net_plotter import plot_model

plt.rcParams['figure.figsize'] = (8, 5)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def load_classification_data(path):
	df = pd.read_csv(path)
	print(f'\nRaw data:\n{df}')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]
	x_to_encode = x.select_dtypes(exclude=np.number).columns
	classes = y.unique()

	for col in x_to_encode:
		if len(x[col].unique()) > 2:
			one_hot = pd.get_dummies(x[col], prefix=col)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)
		else:  # Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True)

	if len(classes) > 2:
		one_hot = pd.get_dummies(y, prefix='class')
		y = pd.concat([y, one_hot], axis=1)
		y = y.drop(y.columns[0], axis=1)
	else:  # Binary class
		y = pd.get_dummies(y, prefix='class')
		# Ensure dummy column corresponds with 'classes'
		drop_idx = int(y.columns[0].endswith(classes[0]))
		y = y.drop(y.columns[drop_idx], axis=1)
		if y.iloc[0][0] == 1:
			# classes[0] = no/false/0
			classes = classes[::-1]

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}\n')

	# Standardise x (numeric features only)
	numeric_feature_indices = [idx for idx, f in enumerate(x.columns) if f not in x_to_encode]
	x, y = x.to_numpy().astype(float), np.squeeze(y.to_numpy().astype(int))
	# Train:validation:test ratio of 0.7:0.2:0.1
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, stratify=y, random_state=1)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.78, stratify=y_train, random_state=1)
	training_mean = np.mean(x_train[:, numeric_feature_indices], axis=0)
	training_std = np.std(x_train[:, numeric_feature_indices], axis=0)
	x_train[:, numeric_feature_indices] = (x_train[:, numeric_feature_indices] - training_mean) / training_std
	x_val[:, numeric_feature_indices] = (x_val[:, numeric_feature_indices] - training_mean) / training_std
	x_test[:, numeric_feature_indices] = (x_test[:, numeric_feature_indices] - training_mean) / training_std

	return classes, x_train, y_train, x_val, y_val, x_test, y_test

def load_regression_data(path):
	df = pd.read_csv(path)
	print(f'\nRaw data:\n{df}')

	x, y = df.iloc[:, :-1], df.iloc[:, -1]
	x_to_encode = x.select_dtypes(exclude=np.number).columns

	for col in x_to_encode:
		if len(x[col].unique()) > 2:
			one_hot = pd.get_dummies(x[col], prefix=col)
			x = pd.concat([x, one_hot], axis=1).drop(col, axis=1)
		else:  # Binary feature
			x[col] = pd.get_dummies(x[col], drop_first=True)

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}\n')

	# Standardise x (numeric features only)
	numeric_feature_indices = [idx for idx, f in enumerate(x.columns) if f not in x_to_encode]
	x, y = x.to_numpy().astype(float), y.to_numpy().astype(float)
	# Train:validation:test ratio of 0.7:0.2:0.1
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=1)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.78, random_state=1)
	training_mean = np.mean(x_train[:, numeric_feature_indices], axis=0)
	training_std = np.std(x_train[:, numeric_feature_indices], axis=0)
	x_train[:, numeric_feature_indices] = (x_train[:, numeric_feature_indices] - training_mean) / training_std
	x_val[:, numeric_feature_indices] = (x_val[:, numeric_feature_indices] - training_mean) / training_std
	x_test[:, numeric_feature_indices] = (x_test[:, numeric_feature_indices] - training_mean) / training_std

	return x_train, y_train, x_val, y_val, x_test, y_test

def confusion_matrix(predictions, actual):
	n_classes = len(np.unique(actual))
	conf_mat = np.zeros((n_classes, n_classes)).astype(int)

	for a, p in zip(actual, predictions):
		conf_mat[a][p] += 1

	f1 = f1_score(actual, predictions, average='binary' if n_classes == 2 else 'weighted')

	return conf_mat, f1

def plot_confusion_matrices(train_conf_mat, train_f1, test_conf_mat, test_f1):
	_, (train, test) = plt.subplots(ncols=2, sharex=True, sharey=True)
	train.matshow(train_conf_mat, cmap=plt.cm.plasma)
	test.matshow(test_conf_mat, cmap=plt.cm.plasma)
	train.xaxis.set_ticks_position('bottom')
	test.xaxis.set_ticks_position('bottom')
	for (j, i), val in np.ndenumerate(train_conf_mat):
		train.text(x=i, y=j, s=val, ha='center', va='center')
		test.text(x=i, y=j, s=test_conf_mat[j][i], ha='center', va='center')
	train.set_xlabel('Predictions')
	train.set_ylabel('Actual')
	train.set_title(f'Training Confusion Matrix\nF1 score: {train_f1}')
	test.set_title(f'Test Confusion Matrix\nF1 score: {test_f1}')
	plt.show()

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def main():
	np.random.seed(1)
	set_seed(1)

	task_choice = input('\nEnter B for binary classification,'
		+ '\nM for multiclass classification,'
		+ '\nor R for regression\n>>> ').upper()
	print()

	match task_choice:
		case 'B':
			dataset_choice = input('Enter 1 for banknote dataset,'
				+ '\n2 for breast tumour dataset,'
				+ '\n3 for pulsar dataset,'
				+ '\nor 4 for Titanic dataset\n>>> ')
		case 'M':
			dataset_choice = input('Enter I for iris dataset,'
				+ '\nor W for wine dataset\n>>> ').upper()
		case 'R':
			dataset_choice = input('Enter B for Boston housing dataset,'
				+ '\nC for car value dataset,'
				+ '\nor M for medical insurance dataset\n>>> ').upper()
		case _:
			print('Bad choice')
			return

	match task_choice + dataset_choice:
		case 'B1': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\banknoteData.csv'
		case 'B2': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\breastTumourData.csv'
		case 'B3': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\pulsarData.csv'
		case 'B4': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\titanicData.csv'
		case 'MI': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\irisData.csv'
		case 'MW': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\wineData.csv'
		case 'RB': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\bostonData.csv'
		case 'RC': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\carValueData.csv'
		case 'RM': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\medicalInsuranceData.csv'
		case _:
			print('Bad choice')
			return

	classes = None
	if task_choice in 'BM':
		classes, x_train, y_train, x_val, y_val, x_test, y_test = load_classification_data(path)
	else:
		x_train, y_train, x_val, y_val, x_test, y_test = load_regression_data(path)

	# ----- Build model ----- #

	n_features = x_train.shape[1]

	match task_choice + dataset_choice:
		case 'B1' | 'B2' | 'B3':  # Banknote, breast tumour, or pulsar dataset
			model = Sequential([
				Dense(units=8, input_shape=(n_features,), activation='relu'),
				Dense(units=1, input_shape=(n_features,), activation='sigmoid')
			])
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		case 'B4':  # Titanic dataset
			model = Sequential([
				Dense(units=8, input_shape=(n_features,), activation='relu'),
				Dropout(0.1),
				Dense(units=1, input_shape=(n_features,), activation='sigmoid')
			])
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		case 'MI':  # Iris dataset
			model = Sequential([
				Dense(units=64, input_shape=(n_features,), activation='relu'),
				Dense(units=64, input_shape=(n_features,), activation='relu'),
				Dense(units=len(classes), input_shape=(n_features,), activation='softmax')
			])
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		case 'MW':  # Wine dataset
			model = Sequential([
				Dense(units=16, input_shape=(n_features,), activation='relu'),
				Dense(units=len(classes), input_shape=(n_features,), activation='softmax')
			])
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		case 'RB':  # Boston housing dataset
			model = Sequential([
				Dense(units=256, input_shape=(n_features,), activation='relu'),
				Dropout(0.1),
				Dense(units=1, input_shape=(n_features,), activation='linear')
			])
			model.compile(loss='mse', metrics=[RootMeanSquaredError()])
		case 'RC':  # Car value dataset
			model = Sequential([
				Dense(units=256, input_shape=(n_features,), activation='relu'),
				Dense(units=256, input_shape=(n_features,), activation='relu'),
				Dense(units=1, input_shape=(n_features,), activation='linear')
			])
			model.compile(loss='mse', metrics=[RootMeanSquaredError()])
		case _:  # Medical insurance dataset
			model = Sequential([
				Dense(units=4096, input_shape=(n_features,), activation='relu'),
				Dense(units=1, input_shape=(n_features,), activation='linear')
			])
			model.compile(loss='mse', metrics=[RootMeanSquaredError()])

	model.build(input_shape=(n_features,))
	model.summary()
	plot_model(model)

	# ----- Training ----- #

	print('\nTraining...')

	early_stopping = EarlyStopping(monitor='val_loss',
		min_delta=0,
		patience=5,
		restore_best_weights=True)

	history = model.fit(x_train, y_train,
		epochs=100,
		validation_data=(x_val, y_val),
		callbacks=[early_stopping],
		verbose=0)

	# Plot loss and accuracy throughout training

	if task_choice in 'BM':
		_, (ax_loss, ax_accuracy) = plt.subplots(nrows=2, sharex=True)
		ax_loss.plot(history.history['loss'], label='Training loss')
		ax_loss.plot(history.history['val_loss'], label='Validation loss')
		ax_accuracy.plot(history.history['accuracy'], label='Training accuracy')
		ax_accuracy.plot(history.history['val_accuracy'], label='Validation accuracy')
		ax_accuracy.set_xlabel('Epoch')
		ax_accuracy.set_ylabel('Accuracy')
		ax_loss.set_ylabel('Categorical cross-entropy loss')
		ax_loss.legend()
		ax_accuracy.legend()
		plt.title('Loss and accuracy during training', y=2.24)
	else:
		plt.plot(history.history['loss'], label='Training MSE')
		plt.plot(history.history['val_loss'], label='Validation MSE')
		plt.xlabel('Epoch')
		plt.ylabel('MSE')
		plt.legend()
		plt.title('MSE during training')
	plt.show()

	# ----- Evaluation ----- #

	print('\nEvaluation\n')

	if task_choice in 'BM':
		test_loss, test_accuracy = model.evaluate(x_test, y_test)
		print('\nTest loss:', test_loss)
		print('Test accuracy:', test_accuracy)
	else:
		test_mse, test_rmse = model.evaluate(x_test, y_test)
		print('\nTest MSE:', test_mse)
		print('Test RMSE:', test_rmse)

	if task_choice == 'R': return

	# ----- Testing ----- #

	print('\nTesting')

	train_predictions = model.predict(x_train)
	test_predictions = model.predict(x_test)

	if task_choice == 'B':
		train_predictions = np.where(train_predictions > 0.5, 1, 0)
		test_predictions = np.where(test_predictions > 0.5, 1, 0)
	else:  # Multiclass
		y_train, y_test = np.argmax(y_train, axis=1), np.argmax(y_test, axis=1)
		train_predictions = np.argmax(train_predictions, axis=1)
		test_predictions = np.argmax(test_predictions, axis=1)

	train_conf_mat, train_f1 = confusion_matrix(train_predictions, y_train)
	test_conf_mat, test_f1 = confusion_matrix(test_predictions, y_test)
	plot_confusion_matrices(train_conf_mat, train_f1, test_conf_mat, test_f1)

if __name__ == '__main__':
	main()
