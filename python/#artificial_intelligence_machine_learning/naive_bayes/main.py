"""
Naive Bayes classification demo

Author: Sam Barba
Created 21/11/2021
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from naive_bayes_classifier import NaiveBayesClassifier


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

	label_encoder = LabelEncoder()
	y = pd.DataFrame(label_encoder.fit_transform(y), columns=['classification'])

	print(f'\nPreprocessed data:\n\n{pd.concat([x, y], axis=1)}')

	x, y = x.to_numpy(dtype=float), y.to_numpy().squeeze()

	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio, stratify=y, random_state=1)

	return x_train, y_train, x_test, y_test, labels


if __name__ == '__main__':
	choice = input(
		'\nEnter 1 to use banknote dataset,'
		'\n2 for breast tumour dataset,'
		'\n3 for glass dataset,'
		'\n4 for iris dataset,'
		'\n5 for mushroom dataset,'
		'\n6 for pulsar dataset,'
		'\n7 for Titanic dataset,'
		'\nor 8 for wine dataset\n>>> '
	)

	match choice:
		case '1': path = 'C:/Users/Sam/Desktop/Projects/datasets/banknote_authentication.csv'
		case '2': path = 'C:/Users/Sam/Desktop/Projects/datasets/breast_tumour_pathology.csv'
		case '3': path = 'C:/Users/Sam/Desktop/Projects/datasets/glass_classification.csv'
		case '4': path = 'C:/Users/Sam/Desktop/Projects/datasets/iris_classification.csv'
		case '5': path = 'C:/Users/Sam/Desktop/Projects/datasets/mushroom_edibility_classification.csv'
		case '6': path = 'C:/Users/Sam/Desktop/Projects/datasets/pulsar_identification.csv'
		case '7': path = 'C:/Users/Sam/Desktop/Projects/datasets/titanic_survivals.csv'
		case _: path = 'C:/Users/Sam/Desktop/Projects/datasets/wine_classification.csv'

	x_train, y_train, x_test, y_test, labels = load_data(path)

	clf = NaiveBayesClassifier()
	clf.fit(x_train, y_train)

	test_pred_probs = clf.predict(x_test)
	test_pred_labels = test_pred_probs.argmax(axis=1)

	# Confusion matrix

	f1 = f1_score(y_test, test_pred_labels, average='binary' if len(labels) == 2 else 'weighted')
	cm = confusion_matrix(y_test, test_pred_labels)
	ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(cmap='Blues')
	plt.title(f'Test confusion matrix\n(F1 score: {f1:.3f})')
	plt.show()

	# ROC curve

	if len(labels) == 2:  # Binary classification
		fpr, tpr, _ = roc_curve(y_test, test_pred_probs[:, 1])  # Assuming 1 is the positive class
		plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
		plt.plot(fpr, tpr)
		plt.axis('scaled')
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC curve')
		plt.show()
