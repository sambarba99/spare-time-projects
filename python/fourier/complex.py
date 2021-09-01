# Complex number for fourier.py
# Author: Sam Barba
# Created 23/09/2021

from math import atan2

class Complex:
	def __init__(self, re, im):
		self.re = re
		self.im = im
		self.freq = 0

	def add(self, other):
		return Complex(self.re + other.re, self.im + other.im)

	def mult(self, other):
		newRe = self.re * other.re - self.im * other.im
		newIm = self.re * other.im + self.im * other.re
		return Complex(newRe, newIm)

	def getAmp(self):
		return (self.re ** 2 + self.im ** 2) ** 0.5

	def getPhase(self):
		return atan2(self.im, self.re)
