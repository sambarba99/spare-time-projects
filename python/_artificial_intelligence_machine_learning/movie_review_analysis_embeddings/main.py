"""
Sentiment Analysis of Movie Reviews with a PyTorch Embedding model

Author: Sam Barba
Created 2024-03-26
"""

from collections import Counter
from pathlib import Path

# import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from _utils.plotting import plot_confusion_matrix, plot_roc_curve, plot_torch_model
from _utils.progress_bar import ProgressBar
from model import MovieReviewClf


# Un-comment if running for first time
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

pd.set_option('display.max_columns', 5)
pd.set_option('display.width', None)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
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
	vocab_size = len(token_to_idx) + 1  # Add 1 for the padding token (0)

	# Convert tokens to indices, cropping sequences to length SEQUENCE_LEN
	df['token_idx'] = df['tokens'].apply(
		lambda tokens: [token_to_idx[t] for idx, t in enumerate(tokens) if idx < SEQUENCE_LEN]
	)

	# Pad shorter sequences with 0s
	df['padded_idx'] = df['token_idx'].apply(lambda idx: idx + [0] * (SEQUENCE_LEN - len(idx)))

	print(f'\nPreprocessed reviews:\n{df}')

	x = list(df['padded_idx'])
	y = pd.get_dummies(df['sentiment'], drop_first=True, dtype=int).to_numpy().squeeze()
	x, y = torch.tensor(x).int(), torch.tensor(y).float()
	labels = sorted(df['sentiment'].unique())

	# Create train/validation/test sets (ratio 0.96:0.02:0.02)
	x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, train_size=0.96, stratify=y, random_state=1)
	x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, train_size=0.5, stratify=y_tmp, random_state=1)

	train_dataset = CustomDataset(x_train, y_train)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

	return train_loader, x_val, y_val, x_test, y_test, labels, vocab_size


if __name__ == '__main__':
	# Load data

	train_loader, x_val, y_val, x_test, y_test, labels, vocab_size = load_data()

	# Define and train model

	model = MovieReviewClf(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
	plot_torch_model(model, (SEQUENCE_LEN,), dtypes=[torch.long], device=DEVICE)

	loss_func = torch.nn.BCEWithLogitsLoss()

	if Path('./model.pth').exists():
		model.load_state_dict(torch.load('./model.pth', map_location=DEVICE))
	else:
		print('\n----- TRAINING -----\n')

		optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
		early_stopping = EarlyStopping(target=model, patience=20, mode='max')

		for epoch in range(1, NUM_EPOCHS + 1):
			prog_bar = ProgressBar(train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS}', unit='batch', auto_finish=False)

			for x_train, y_train in prog_bar:
				logits = model(x_train.to(DEVICE)).squeeze()
				loss = loss_func(logits, y_train.to(DEVICE))

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

				prog_bar.set_postfix(f'loss={loss.item():.4f}')

			with torch.inference_mode():
				logits = model(x_val.to(DEVICE)).squeeze().cpu()
			val_probs = torch.sigmoid(logits)
			val_preds = val_probs.round().detach()
			val_loss = loss_func(logits, y_val).item()
			val_f1 = f1_score(y_val, val_preds)
			prog_bar.finish(f'{val_loss=:.4f}, {val_f1=:.4f}')

			if early_stopping(val_f1):
				early_stopping.print_stop_message()
				break

		early_stopping.restore_best_weights()
		torch.save(model.state_dict(), './model.pth')

	print('\n----- TESTING -----\n')

	with torch.inference_mode():
		logits = model(x_test.to(DEVICE)).squeeze().cpu()
	probs = torch.sigmoid(logits)
	preds = probs.round().detach()

	loss = loss_func(logits, y_test)
	print('Loss:', loss.item())

	f1 = f1_score(y_test, preds)
	plot_confusion_matrix(y_test, preds, labels, f'Test confusion matrix\n(F1 score: {f1:.3f})')

	plot_roc_curve(y_test, probs)
