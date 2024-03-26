"""
Sentiment Analysis of movie reviews via an embedding-based PyTorch model

Author: Sam Barba
Created 26/03/2024
"""

from collections import Counter
import os

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch

from _utils.early_stopping import EarlyStopping
from _utils.model_evaluation_plots import plot_confusion_matrix, plot_roc_curve
from movie_review_classifier import MovieReviewClf


nltk.download('punkt')
nltk.download('stopwords')
pd.set_option('display.max_columns', 5)
pd.set_option('display.width', None)
torch.manual_seed(1)

MAX_SEQ_LENGTH = 200
EMBEDDING_DIM = 256
HIDDEN_DIM = 128
N_EPOCHS = 100
BATCH_SIZE = 64


def load_data(path):
	def preprocess_text(text):
		tokens = word_tokenize(text)
		tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
		return tokens


	df = pd.read_csv(path)
	print(f'\nRaw data:\n{df}')

	# Tokenisation and preprocessing

	stop_words = set(stopwords.words('english'))
	df['tokens'] = df['review'].apply(preprocess_text)

	# Build vocabulary and assign indices to words
	all_words = [word for tokens in df['tokens'] for word in tokens]
	word_counts = Counter(all_words)
	sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
	word_to_idx = {word: idx for idx, word in enumerate(sorted_words, start=1)}
	vocab_size = len(word_to_idx) + 1  # Add 1 for the padding token

	# Convert words to indices in the dataset, and pad sequences to a fixed length
	df['indexed_tokens'] = df['tokens'].apply(lambda tokens: [word_to_idx[word] for word in tokens])
	df['padded_tokens'] = df['indexed_tokens'].apply(lambda indices: indices[:MAX_SEQ_LENGTH] + [0] * (MAX_SEQ_LENGTH - len(indices)))

	print(f'\nPreprocessed reviews:\n{df}')

	# Create train/validation/test sets (ratio 0.7:0.2:0.1)

	labels = sorted(np.unique(df['sentiment'].values))
	x = list(df['padded_tokens'])
	y = pd.get_dummies(df['sentiment'], drop_first=True, dtype=int).to_numpy().squeeze()
	x, y = torch.IntTensor(x), torch.IntTensor(y)

	x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, train_size=0.9, stratify=y, random_state=1)
	x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, train_size=0.78, stratify=y_train_val, random_state=1)

	return labels, vocab_size, x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':
	# 1. Get data

	labels, vocab_size, x_train, y_train, x_val, y_val, x_test, y_test = load_data('C:/Users/Sam/Desktop/projects/datasets/movie_reviews.csv')

	# 2. Define and train model

	model = MovieReviewClf(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
	loss_func = torch.nn.BCELoss()

	print(f'\nModel:\n{model}')

	if os.path.exists('./model.pth'):
		model.load_state_dict(torch.load('./model.pth'))
	else:
		print('\n----- TRAINING -----\n')
		optimiser = torch.optim.Adam(model.parameters())  # LR = 1e-3
		early_stopping = EarlyStopping(patience=10, min_delta=0, mode='max')

		for epoch in range(1, N_EPOCHS + 1):
			for i in range(0, len(x_train), BATCH_SIZE):
				x_batch = x_train[i:i + BATCH_SIZE]
				y_batch = y_train[i:i + BATCH_SIZE]

				y_train_probs = model(x_batch)
				loss = loss_func(y_train_probs, y_batch.float())

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

			with torch.inference_mode():
				y_val_probs = model(x_val)
			y_val_pred = y_val_probs.round()
			val_f1 = f1_score(y_val, y_val_pred)

			if epoch % 5 == 0:
				print(f'Epoch {epoch}/{N_EPOCHS}: val F1 = {val_f1}')

			if early_stopping(val_f1, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './model.pth')

	# 3. Test model (plot confusion matrix and ROC curve)

	with torch.inference_mode():
		y_test_probs = model(x_test)
	y_test_pred = y_test_probs.round()

	test_loss = loss_func(y_test_probs, y_test.float()).item()
	print('\nTest loss:', test_loss)

	f1 = f1_score(y_test, y_test_pred)
	plot_confusion_matrix(y_test, y_test_pred, labels, f'Test confusion matrix\n(F1 score: {f1:.3f})')

	plot_roc_curve(y_test, y_test_probs)
