"""
eXtreme Gradient Boost (XGBoost) demo

Author: Sam Barba
Created 09/03/2024
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from _utils.csv_data_loader import load_csv_classification_data
from _utils.plotting import plot_confusion_matrix, plot_roc_curve
from xgboost import XGBoostClassifier


plt.rcParams['figure.figsize'] = (6, 4)
np.random.seed(1)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', None)


if __name__ == '__main__':
	choice = input(
		'\nEnter 1 for banknote dataset,'
		'\n2 for breast tumour dataset,'
		'\n3 for glass dataset,'
		'\n4 for iris dataset,'
		'\n5 for mushroom dataset,'
		'\n6 for pulsar dataset,'
		'\n7 for Titanic dataset,'
		'\nor 8 for wine dataset\n>>> '
	)

	match choice:
		case '1': path = 'C:/Users/sam/Desktop/projects/datasets/banknote_authenticity.csv'
		case '2': path = 'C:/Users/sam/Desktop/projects/datasets/breast_tumour_pathology.csv'
		case '3': path = 'C:/Users/sam/Desktop/projects/datasets/glass_classification.csv'
		case '4': path = 'C:/Users/sam/Desktop/projects/datasets/iris_classification.csv'
		case '5': path = 'C:/Users/sam/Desktop/projects/datasets/mushroom_edibility_classification.csv'
		case '6': path = 'C:/Users/sam/Desktop/projects/datasets/pulsar_identification.csv'
		case '7': path = 'C:/Users/sam/Desktop/projects/datasets/titanic_survivals.csv'
		case _: path = 'C:/Users/sam/Desktop/projects/datasets/wine_classification.csv'

	x_train, y_train, x_test, y_test, labels, _ = load_csv_classification_data(path, train_size=0.8, test_size=0.2)

	# Fit a model and predict

	clf = XGBoostClassifier(num_estimators=100, max_depth=4, learning_rate=0.1, num_classes=len(labels))
	clf.fit(x_train, y_train)

	test_pred_probs = clf.predict(x_test)
	test_pred_classes = test_pred_probs.argmax(axis=1)

	# Confusion matrix
	f1 = f1_score(y_test, test_pred_classes, average='binary' if len(labels) == 2 else 'weighted')
	plot_confusion_matrix(
		y_test,
		test_pred_classes,
		labels,
		f'Test confusion matrix\n(F1 score: {f1:.3f})',
		x_ticks_rotation=45,
		horiz_alignment='right'
	)

	# ROC curve
	if len(labels) == 2:  # Binary classification
		plot_roc_curve(y_test, test_pred_probs[:, 1])  # Assuming 1 is the positive class
