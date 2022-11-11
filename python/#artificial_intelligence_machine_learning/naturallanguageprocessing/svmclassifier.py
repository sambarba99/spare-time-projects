"""
SVM classifier for NLP demo

Author: Sam Barba
Created 29/09/2022
"""

from numpy import exp, tanh
from sklearn import svm
from sklearn.metrics.pairwise import euclidean_distances

class SVMClassifier:
	def __init__(self):
		self.x_train = None
		self.y_train = None
		self.r = -1
		self.C = 100
		self.gamma = 1
		self.degree = 3
		self.sc = svm.SVC(C=self.C, kernel=self.gaussian, gamma=self.gamma, degree=self.degree)

	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train
		self.sc.fit(self.x_train, self.y_train)

	def linear(self, x, y):
		return x.dot(y.T)

	def poly(self, x, y, gamma=1, r=0.8):  # C = 100
		return (gamma * x.dot(y.T) + r) ** self.degree

	def sigmoid(self, x, y, gamma=1, r=-1):  # C = 100
		k = gamma * (x.dot(y.T)) + r
		return tanh(k)

	def gaussian(self, x, y, gamma=0.58):  # C = 100
		k = euclidean_distances(x, y) ** 2
		k *= -1 / (gamma * 2)
		return exp(k)

	def my_kernel(self, x, y):
		# k1 = self.sigmoid(x, y)
		k2 = self.gaussian(x, y)
		k3 = self.poly(x, y)
		return 0.5 * k2 * 0.5 * k3

	def predict(self, x):
		return self.sc.predict(x)
