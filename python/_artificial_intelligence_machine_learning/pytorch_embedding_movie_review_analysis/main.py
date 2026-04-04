"""
Sentiment Analysis of movie reviews via an embedding-based PyTorch model

Author: Sam Barba
Created 26/03/2024
"""

from collections import Counter
from pathlib import Path

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
from _utils.plotting import plot_confusion_matrix, plot_roc_curve, plot_torch_model
from model import MovieReviewClf

# Un-comment if running for first time
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

pd.set_option('display.max_columns', 5)
pd.set_option('display.width', None)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

SEQUENCE_LEN = 250
EMBEDDING_DIM = 256
HIDDEN_DIM = 128
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
NUM_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data():
	df = pd.read_csv('C:/Users/sam/Desktop/projects/datasets/imdb_movie_reviews.csv')
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

	df['token_idx'] = df['tokens'].apply(
		lambda tokens: [token_to_idx[t] for idx, t in enumerate(tokens) if idx < SEQUENCE_LEN]
	)
	df['padded_idx'] = df['token_idx'].apply(lambda idx: idx + [0] * (SEQUENCE_LEN - len(idx)))

	print(f'\nPreprocessed reviews:\n{df}')

	# Create train/validation/test sets (ratio 0.96:0.02:0.02)

	x = list(df['padded_idx'])
	y = pd.get_dummies(df['sentiment'], drop_first=True, dtype=int).to_numpy().squeeze()
	x, y = torch.tensor(x).int(), torch.tensor(y).float()
	labels = sorted(df['sentiment'].unique())

	x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, train_size=0.98, stratify=y, random_state=1)
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

	model = MovieReviewClf(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
	plot_torch_model(model, (SEQUENCE_LEN,), device=DEVICE)

	loss_func = torch.nn.BCEWithLogitsLoss()

	if Path('./model.pth').exists():
		model.load_state_dict(torch.load('./model.pth', map_location=DEVICE))
	else:
		print('\n----- TRAINING -----\n')
		optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
		early_stopping = EarlyStopping(model=model, patience=20, mode='max', track_best_weights=True)

		for epoch in range(1, NUM_EPOCHS + 1):
			progress_bar = tqdm(range(len(train_loader)), unit='batches', ascii=True)

			for x_train, y_train in train_loader:
				progress_bar.update()
				progress_bar.set_description(f'Epoch {epoch}/{NUM_EPOCHS}')

				y_train_logits = model(x_train.to(DEVICE)).squeeze()
				loss = loss_func(y_train_logits, y_train.to(DEVICE))

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

				progress_bar.set_postfix_str(f'loss={loss.item():.4f}')

			with torch.inference_mode():
				y_val_logits = model(x_val.to(DEVICE)).squeeze().cpu()
			y_val_probs = torch.sigmoid(y_val_logits)
			y_val_pred = y_val_probs.round().detach()
			val_loss = loss_func(y_val_logits, y_val).item()
			val_f1 = f1_score(y_val, y_val_pred)
			progress_bar.set_postfix_str(f'val_loss={val_loss:.4f}, val_F1={val_f1:.4f}')
			progress_bar.close()

			if early_stopping(val_f1):
				break

		early_stopping.restore_best_weights()
		torch.save(model.state_dict(), './model.pth')

	# Test model (plot confusion matrix and ROC curve)

	with torch.inference_mode():
		y_test_logits = model(x_test.to(DEVICE)).squeeze().cpu()
	y_test_probs = torch.sigmoid(y_test_logits)
	y_test_pred = y_test_probs.round().detach()

	test_loss = loss_func(y_test_logits, y_test).item()
	print('\nTest loss:', test_loss)

	f1 = f1_score(y_test, y_test_pred)
	plot_confusion_matrix(y_test, y_test_pred, labels, f'Test confusion matrix\n(F1 score: {f1:.3f})')

	plot_roc_curve(y_test, y_test_probs)
