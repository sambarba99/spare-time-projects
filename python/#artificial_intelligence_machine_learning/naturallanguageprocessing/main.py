"""
NLP demo

Sentiment analysis of movie reviews is performed by an SVM and a boosting-based algorithm.
NLTK (Natural Language Toolkit) is used.

Author: Sam Barba
Created 29/09/2022
"""

import re

import nltk
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from gradientboost import GradientBoost
from svmclassifier import SVMClassifier

STOP_WORDS = ['!', '(', ')', '*', '-', '..', '/', ':', ';', '<', '>', '?', '``']

def extract_bag_of_words(raw_data, train_test_ratio=0.8):
	stemmer = LancasterStemmer()

	cleaned_data = []

	for row in tqdm(raw_data, desc='Processing raw data', ascii=True):
		review, sentiment = row.rsplit(',', maxsplit=1)
		review = re.sub(r'["\'`]', '', review)
		word_tokens = word_tokenize(re.sub(r'[.,]', '', review))
		stemmed_doc = [stemmer.stem(w) for w in word_tokens]
		filtered_review = ' '.join([word for word in stemmed_doc if word.lower() not in STOP_WORDS])
		cleaned_data.append([filtered_review, sentiment])

	x, y = np.array(cleaned_data).astype(str).T

	# Convert positive/negative to 1/-1 respectively
	y = np.where(y == 'positive', 1, -1)

	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_test_ratio, stratify=y)

	vectorizer = TfidfVectorizer()
	x_train = vectorizer.fit_transform(x_train).toarray()
	x_test = vectorizer.transform(x_test).toarray()

	print('\nx_train shape:', x_train.shape)
	print('y_train shape:', y_train.shape)
	print('x_test shape:', x_test.shape)
	print('y_test shape:', y_test.shape)

	return x_train, y_train, x_test, y_test

if __name__ == '__main__':
	# Uncomment and run this if not done already (just needs to run once)
	# nltk.download('punkt')

	with open(r'C:\Users\Sam Barba\Desktop\Programs\datasets\movieReviews.csv', 'r', encoding='UTF-8') as file:
		data = file.read().splitlines()[1:]

	x_train, y_train, x_test, y_test = extract_bag_of_words(data)

	# Test SVM
	svm = SVMClassifier()
	svm.fit(x_train, y_train)
	y_pred = svm.predict(x_test)
	print(f'\nSVM F1 score: {f1_score(y_test, y_pred)}\n')

	# Test boosting classifier
	gb = GradientBoost(learning_rate=0.5, n_trees=50, max_depth=2)
	gb.fit(x_train, y_train)
	y_pred = gb.predict(x_test)
	print('\n\nBoosting F1 score:', f1_score(y_test, y_pred))
