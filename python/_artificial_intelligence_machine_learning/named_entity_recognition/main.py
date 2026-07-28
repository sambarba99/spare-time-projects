"""
Named Entity Recognition with a PyTorch embedding/LSTM model

Author: Sam Barba
Created 2026-07-28
"""

from collections import Counter
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader

from _utils.early_stopping import EarlyStopping
from _utils.plotting import plot_confusion_matrix, plot_torch_model
from _utils.progress_bar import ProgressBar
from model import NERClf


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
EMBEDDING_DIM = 64
HIDDEN_DIM = 32
BATCH_SIZE = 128
NUM_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def encode(split):
	"""Dataset encoding"""

	encoded = []

	for sample in dataset[split]:
		tokens = [token2idx.get(w, token2idx[UNK_TOKEN]) for w in sample['tokens']]
		tags = sample['ner_tags']
		encoded.append((tokens, tags))

	return encoded


def collate(batch):
	"""Data padding"""

	xs, ys = zip(*batch)
	max_len = max(len(x) for x in xs)

	x_batch, y_batch = [], []

	for x, y in zip(xs, ys):
		pad = max_len - len(x)
		x_batch.append(x + [0] * pad)
		y_batch.append(y + [-100] * pad)  # -100 label is ignored by CrossEntropyLoss

	return torch.tensor(x_batch), torch.tensor(y_batch)


def evaluate(loader):
	pred_labels, true_labels = [], []

	with torch.inference_mode():
		for x, y in loader:
			logits = model(x.to(DEVICE))
			preds = logits.argmax(dim=-1)

			for pred_seq, true_seq in zip(preds, y):
				for p, t in zip(pred_seq.tolist(), true_seq.tolist()):
					if t == -100:
						continue  # Ignore padding
					pred_labels.append(idx2tag[p])
					true_labels.append(idx2tag[t])

	f1 = f1_score(true_labels, pred_labels, average='macro')

	return true_labels, pred_labels, f1


if __name__ == '__main__':
	dataset = load_dataset('lhoestq/conll2003')

	# Build vocabulary and assign indices to tokens

	token_counts = Counter()
	for sample in dataset['train']:
		token_counts.update(sample['tokens'])

	token2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}

	for token in token_counts:
		token2idx[token] = len(token2idx)

	idx2token = {v: k for k, v in token2idx.items()}

	# Tag mappings

	ner_feature = dataset['train'].features['ner_tags'].feature

	if hasattr(ner_feature, 'names'):
		tag_names = ner_feature.names
	else:
		tag_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

	tag2idx = {tag: i for i, tag in enumerate(tag_names)}
	idx2tag = {i: tag for tag, i in tag2idx.items()}
	# print(idx2tag)

	test_data = encode('test')
	test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate)

	model = NERClf(
		vocab_size=len(token2idx),
		embedding_dim=EMBEDDING_DIM,
		hidden_dim=HIDDEN_DIM,
		num_tags=len(tag_names)
	).to(DEVICE)
	plot_torch_model(model, (50,), dtypes=[torch.long], device=DEVICE)

	if Path('./model.pth').exists():
		model.load_state_dict(torch.load('./model.pth', map_location=DEVICE))
	else:
		print('\n----- TRAINING -----\n')

		train_data = encode('train')
		val_data = encode('validation')
		train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
		val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate)
		loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
		optimiser = torch.optim.Adam(model.parameters())  # LR = 1e-3
		early_stopping = EarlyStopping(target=model, patience=10, mode='max')

		# Print some training samples
		for i in range(4):
			tokens, tags = train_data[i]
			print(f'Sample {i + 1}')
			print('Tokens:', [idx2token[t] for t in tokens])
			print('  Tags:', [idx2tag[t] for t in tags], end='\n\n')

		for epoch in range(1, NUM_EPOCHS + 1):
			prog_bar = ProgressBar(train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS}', unit='batch', auto_finish=False)
			model.train()

			for x_train, y_train in prog_bar:
				logits = model(x_train.to(DEVICE))
				loss = loss_func(logits.view(-1, len(tag2idx)), y_train.view(-1).to(DEVICE))

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

				prog_bar.set_postfix(f'loss={loss.item():.4f}')

			model.eval()
			*_, val_f1 = evaluate(val_loader)
			prog_bar.finish(f'{val_f1=:.4f}')

			if early_stopping(val_f1):
				early_stopping.print_stop_message()
				break

		early_stopping.restore_best_weights()
		torch.save(model.state_dict(), './model.pth')

	model.eval()
	true_labels, pred_labels, test_f1 = evaluate(test_loader)
	plot_confusion_matrix(
		true_labels,
		pred_labels,
		sorted(tag_names),
		f'Test confusion matrix\n(F1 score: {test_f1:.3f})',
		x_ticks_rotation=45
	)
