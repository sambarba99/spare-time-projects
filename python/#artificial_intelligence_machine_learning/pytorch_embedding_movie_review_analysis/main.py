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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from early_stopping import EarlyStopping

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


class ReviewClassifier(nn.Module):
	def __init__(self, *, vocab_size):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
		self.fc1 = nn.Linear(EMBEDDING_DIM, HIDDEN_DIM)
		self.fc2 = nn.Linear(HIDDEN_DIM, 1)
		self.leaky_relu = nn.LeakyReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		embedded = self.leaky_relu(self.embedding(x))
		pooled = torch.mean(embedded, dim=1)
		fc1_out = self.leaky_relu(self.fc1(pooled))
		fc2_out = self.sigmoid(self.fc2(fc1_out))
		return fc2_out.squeeze()


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
	y = pd.get_dummies(df['sentiment'], drop_first=True).astype(int).to_numpy().squeeze()
	x, y = torch.tensor(x, dtype=torch.int), torch.from_numpy(y)

	x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, train_size=0.9, stratify=y, random_state=1)
	x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, train_size=0.78, stratify=y_train_val, random_state=1)

	return labels, vocab_size, x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == '__main__':
	# 1. Get data

	labels, vocab_size, x_train, y_train, x_val, y_val, x_test, y_test = load_data('C:/Users/Sam/Desktop/Projects/datasets/movie_reviews.csv')

	# 2. Define and train model

	model = ReviewClassifier(vocab_size=vocab_size)
	loss_func = nn.BCELoss()

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
	cm = confusion_matrix(y_test, y_test_pred)
	ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(cmap='Blues')
	plt.title(f'Test confusion matrix\n(F1 score: {f1:.3f})')
	plt.show()

	fpr, tpr, _ = roc_curve(y_test, y_test_probs)
	plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
	plt.plot(fpr, tpr)
	plt.axis('scaled')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve')
	plt.show()
