"""
Logistic regression using PCA

Author: Sam Barba
Created 10/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from logistic_regressor import LogisticRegressor


plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)


def load_data(path, train_test_ratio=0.8):
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

	y = pd.get_dummies(y, prefix='class', drop_first=True).astype(int)

	print(f'\nPreprocessed data:\n\n{pd.concat([x, y], axis=1)}')

	x, y = x.to_numpy(), y.to_numpy().squeeze()

	pca = PCA(n_components=2)
	scaler = MinMaxScaler()
	x = pca.fit_transform(scaler.fit_transform(x))

	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio, stratify=y, random_state=1)

	return x_train, y_train, x_test, y_test, labels


if __name__ == '__main__':
	choice = input(
		'\nEnter 1 to use banknote dataset,'
		'\n2 for breast tumour dataset,'
		'\nor 3 for mushroom dataset\n>>> '
	)

	match choice:
		case '1': path = 'C:/Users/Sam/Desktop/Projects/datasets/banknote_authentication.csv'
		case '2': path = 'C:/Users/Sam/Desktop/Projects/datasets/breast_tumour_pathology.csv'
		case _: path = 'C:/Users/Sam/Desktop/Projects/datasets/mushroom_edibility_classification.csv'

	x_train, y_train, x_test, y_test, labels = load_data(path)

	regressor = LogisticRegressor(labels)
	regressor.fit(x_train, y_train)

	test_pred_probs = regressor.predict(x_test)
	test_pred_labels = test_pred_probs.round()

	# Confusion matrix

	cm = confusion_matrix(y_test, test_pred_labels)
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
	f1 = f1_score(y_test, test_pred_labels)

	disp.plot(cmap='plasma')
	plt.title(f'Test onfusion matrix\n(F1 score: {f1})')
	plt.show()

	# ROC curve

	fpr, tpr, _ = roc_curve(y_test, test_pred_probs)
	plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
	plt.plot(fpr, tpr)
	plt.axis('scaled')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve')
	plt.show()

	# Cost history

	plt.plot(regressor.cost_history)
	plt.xlabel('Training iteration')
	plt.ylabel('Cost')
	plt.title('Cost during training')
	plt.show()
