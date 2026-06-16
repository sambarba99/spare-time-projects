"""
Support Vector Machine (SVM) demo

Author: Sam Barba
Created 2024-03-13
"""

from sklearn.datasets import make_blobs

from svm import SVM


if __name__ == '__main__':
	x, y = make_blobs(n_samples=200, centers=2, random_state=7)

	clf = SVM()
	clf.fit(x, y)
