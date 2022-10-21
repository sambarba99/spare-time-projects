"""
PCA demo

Author: Sam Barba
Created 10/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pca import transform

plt.rcParams['figure.figsize'] = (7, 7)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)

# ---------------------------------------------------------------------------------------------------- #
# --------------------------------------------  FUNCTIONS  ------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

def load_data(path):
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

	print(f'\nCleaned data:\n{pd.concat([x, y], axis=1)}')

	x, y = x.to_numpy().astype(float), y.to_numpy().astype(int)
	x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))  # Normalise

	return classes, x, y

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	choice = input('Enter 1 to use banknote dataset,'
		+ '\n2 for breast tumour dataset,'
		+ '\n3 for iris dataset,'
		+ '\n4 for pulsar dataset,'
		+ '\n5 for Titanic dataset,'
		+ '\nor 6 for wine dataset\n>>> ')

	match choice:
		case '1': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\banknoteData.csv'
		case '2': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\breastTumourData.csv'
		case '3': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\irisData.csv'
		case '4': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\pulsarData.csv'
		case '5': path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\titanicData.csv'
		case _: path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\wineData.csv'

	classes, x, y = load_data(path)
	x_transform, new_variability = transform(x)

	if len(classes) == 2:
		y = np.squeeze(y)
		plt.scatter(*x_transform[y == 0].T, alpha=0.7, label=classes[0])
		plt.scatter(*x_transform[y == 1].T, alpha=0.7, label=classes[1])
	else:
		for idx, class_one_hot in enumerate(sorted(np.unique(y, axis=0), key=str, reverse=True)):
			# E.g. for iris dataset, class_one_hot will iterate through [1 0 0], [0 1 0], [0 0 1]
			# This means that 'classes' will be indexed correctly with 'idx'
			class_indices = np.all(y == class_one_hot, axis=1)
			plt.scatter(*x_transform[class_indices].T, alpha=0.7, label=classes[idx])

	plt.xlabel('Principal component 1')
	plt.ylabel('Principal component 2')
	plt.title(f'Shape of x: {x.shape}\nShape of PCA transform: {x_transform.shape}'
		f'\nCaptured variability: {new_variability}')
	plt.legend()
	plt.show()
