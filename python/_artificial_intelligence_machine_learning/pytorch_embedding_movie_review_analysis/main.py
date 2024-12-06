"""
Sentiment Analysis of movie reviews via an embedding-based PyTorch model

Author: Sam Barba
Created 26/03/2024
"""

from collections import Counter
import os

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from _utils.plotting import plot_torch_model, plot_confusion_matrix, plot_roc_curve
from model import MovieReviewClf

# Un-comment if running for first time
# nltk.download('punkt')
# nltk.download('stopwords')

pd.set_option('display.max_columns', 5)
pd.set_option('display.width', None)
torch.manual_seed(1)

SEQUENCE_LEN = 300
EMBEDDING_LEN = 256
HIDDEN_LEN = 128
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
NUM_EPOCHS = 50


def load_data():
	df = pd.read_csv('C:/Users/Sam/Desktop/projects/datasets/imdb_movie_reviews.csv')
	print(f'\nRaw data:\n{df}')
	print('\nPreprocessing...')

	# Tokenisation and stopword removal

	stop_words = set(stopwords.words('english'))
	df['tokens'] = df['review'].apply(word_tokenize)
	df['tokens'] = df['tokens'].apply(
		lambda tokens: [t.lower() for t in tokens if t.isalnum() and t.lower() not in stop_words]
	)

	# Build vocabulary and assign indices to tokens

	all_tokens = [t for tokens in df['tokens'] for t in tokens]
	token_counts = Counter(all_tokens)
	sorted_tokens = sorted(token_counts, key=token_counts.get, reverse=True)
	token_to_idx = {t: idx for idx, t in enumerate(sorted_tokens, start=1)}
	vocab_size = len(token_to_idx) + 1  # Add 1 for the padding token

	# Convert tokens to indices, and crop/pad sequences to length SEQUENCE_LEN

	df['indexed_tokens'] = df['tokens'].apply(lambda tokens: [token_to_idx[t] for t in tokens])
	df['padded_tokens'] = df['indexed_tokens'].apply(
		lambda indices: indices[:SEQUENCE_LEN] + [0] * (SEQUENCE_LEN - len(indices))
	)

	print(f'\nPreprocessed reviews:\n{df}')

	# Create train/validation/test sets (ratio 0.96:0.02:0.02)

	x = list(df['padded_tokens'])
	y = pd.get_dummies(df['sentiment'], drop_first=True, dtype=int).to_numpy().squeeze()
	x, y = torch.tensor(x).int(), torch.tensor(y).float()
	labels = sorted(df['sentiment'].unique())

	x_train_val, x_test, y_train_val, y_test = train_test_split(
		x, y, train_size=0.98, stratify=y, random_state=1
	)
	x_train, x_val, y_train, y_val = train_test_split(
		x_train_val, y_train_val, train_size=0.98, stratify=y_train_val, random_state=1
	)

	train_dataset = CustomDataset(x_train, y_train)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

	return train_loader, x_val, y_val, x_test, y_test, labels, vocab_size


if __name__ == '__main__':
	# Prepare data

	train_loader, x_val, y_val, x_test, y_test, labels, vocab_size = load_data()

	# Define and train model

	model = MovieReviewClf(
		vocab_size=vocab_size,
		embedding_len=EMBEDDING_LEN,
		hidden_len=HIDDEN_LEN
	).cpu()
	plot_torch_model(model, (SEQUENCE_LEN,))

	loss_func = torch.nn.BCEWithLogitsLoss()

	if os.path.exists('./model.pth'):
		model.load_state_dict(torch.load('./model.pth'))
	else:
		print('\n----- TRAINING -----\n')
		optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
		early_stopping = EarlyStopping(patience=5, min_delta=0, mode='max')

		for epoch in range(1, NUM_EPOCHS + 1):
			progress_bar = tqdm(range(len(train_loader)), unit='batches', ascii=True)

			for x_train, y_train in train_loader:
				progress_bar.update()
				progress_bar.set_description(f'Epoch {epoch}/{NUM_EPOCHS}')

				y_train_logits = model(x_train).squeeze()
				loss = loss_func(y_train_logits, y_train)

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

				progress_bar.set_postfix_str(f'loss={loss.item():.4f}')

			with torch.inference_mode():
				y_val_logits = model(x_val).squeeze()
			y_val_probs = torch.sigmoid(y_val_logits)
			y_val_pred = y_val_probs.round().detach()
			val_loss = loss_func(y_val_logits, y_val).item()
			val_f1 = f1_score(y_val, y_val_pred)
			progress_bar.set_postfix_str(f'val_loss={val_loss:.4f}, val_F1={val_f1:.4f}')
			progress_bar.close()

			if early_stopping(val_f1, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './model.pth')

	# Test model (plot confusion matrix and ROC curve)

	with torch.inference_mode():
		y_test_logits = model(x_test).squeeze()
	y_test_probs = torch.sigmoid(y_test_logits)
	y_test_pred = y_test_probs.round().detach()

	test_loss = loss_func(y_test_logits, y_test).item()
	print('\nTest loss:', test_loss)

	f1 = f1_score(y_test, y_test_pred)
	plot_confusion_matrix(y_test, y_test_pred, labels, f'Test confusion matrix\n(F1 score: {f1:.3f})')

	plot_roc_curve(y_test, y_test_probs)
