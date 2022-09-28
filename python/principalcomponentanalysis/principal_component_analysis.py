"""
PCA demo

Author: Sam Barba
Created 10/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
from pca import PCA

plt.rcParams['figure.figsize'] = (7, 7)

# ---------------------------------------------------------------------------------------------------- #
# ----------------------------------------------  MAIN  ---------------------------------------------- #
# ---------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
	choice = input('Enter B to use breast tumour dataset,'
		+ '\nI for iris dataset,'
		+ '\nP for pulsar dataset,'
		+ '\nT for Titanic dataset,'
		+ '\nor W for wine dataset\n>>> ').upper()

	match choice:
		case 'B':
			path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\breastTumourData.txt'
			classes = ['malignant', 'benign']
		case 'I':
			path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\irisData.txt'
			classes = ['setosa', 'versicolor', 'virginica']
		case 'P':
			path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\pulsarData.txt'
			classes = ['not pulsar', 'pulsar']
		case 'T':
			path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\titanicData.txt'
			classes = ['did not survive', 'survived']
		case _:
			path = r'C:\Users\Sam Barba\Desktop\Programs\datasets\wineData.txt'
			classes = ['class 0', 'class 1', 'class 2']

	data = np.genfromtxt(path, dtype=str, delimiter='\n')
	# Skip header and convert to floats
	data = [row.split() for row in data[1:]]
	data = np.array(data).astype(float)

	x, y = data[:, :-1], data[:, -1].astype(int)

	pca = PCA()
	x_transform, new_variability = pca.transform(x)

	for idx, class_label in enumerate(np.unique(y)):
		plt.scatter(*x_transform[y == class_label].T, alpha=0.7, label=classes[idx])
	plt.xlabel('Principal component 1')
	plt.ylabel('Principal component 2')
	plt.title(f'Shape of x: {x.shape}\nShape of PCA transform: {x_transform.shape}'
		f'\nCaptured variability: {new_variability:.3f}')
	plt.legend()
	plt.show()
