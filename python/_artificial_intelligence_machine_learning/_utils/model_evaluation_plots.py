"""
Model evaluation plotting functionality

Author: Sam Barba
Created 26/03/2024
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve


def plot_confusion_matrix(y_test, test_pred_labels, labels, title):
	cm = confusion_matrix(y_test, test_pred_labels)
	ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(cmap='Blues')
	plt.title(title)
	plt.show()


def plot_roc_curve(y_test, test_pred_probs, title='ROC curve'):
	fpr, tpr, _ = roc_curve(y_test, test_pred_probs)
	plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
	plt.plot(fpr, tpr)
	plt.axis('scaled')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(title)
	plt.show()
