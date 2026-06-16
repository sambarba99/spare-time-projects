"""
Adversarial patch generation

Author: Sam Barba
Created 2026-06-17
"""

from pathlib import Path

import torch

from _utils.early_stopping import EarlyStopping
from _utils.progress_bar import ProgressBar
from utils import *


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

LEARNING_RATE = 0.01
NUM_EPOCHS = 100


def train_patch(patch_size, target_class_name=None):
	if target_class_name:
		print(f"\nTraining {patch_size}x{patch_size} patch (target class = '{target_class_name}')\n")
	else:
		print(f'\nTraining {patch_size}x{patch_size} patch (untargeted)\n')

	torch.manual_seed(1)
	torch.cuda.manual_seed_all(1)

	patch_logits = torch.nn.Parameter(0.01 * torch.randn(3, patch_size, patch_size, device=DEVICE))
	target_label = imagenet_labels.index(target_class_name) if target_class_name else None
	loss_func = torch.nn.CrossEntropyLoss()
	optimiser = torch.optim.Adam([patch_logits], lr=LEARNING_RATE)
	early_stopping = EarlyStopping(target=patch_logits, patience=10, mode='max')

	prog_bar = ProgressBar(range(NUM_EPOCHS), desc='Training', unit='epoch')

	for _ in prog_bar:
		for x_train, y_train in train_loader:
			x_train = x_train.to(DEVICE)
			y_train = y_train.to(DEVICE)

			# Expectation over transformations/translations (EOT):
			# sample a random patch location to optimise expected attack performance over translations
			patched = apply_patch(x_train, patch_logits)
			logits = model(patched)

			if target_label is None:
				# Untargeted patch objective: maximise loss on the true label (i.e. negate cross-entropy loss)
				loss = -loss_func(logits, y_train)
			else:
				# Targeted patch objective: minimise loss on the target label
				target = torch.full_like(y_train, target_label)
				loss = loss_func(logits, target)

			optimiser.zero_grad()
			loss.backward()
			optimiser.step()

		successes = total = 0

		with torch.inference_mode():
			for x_val, y_val in val_loader:
				patched = apply_patch(x_val.to(DEVICE), patch_logits, validation=True)
				logits = model(patched).cpu()
				preds = logits.argmax(dim=1)

				if target_label is None:
					# Untargeted attack success rate: prediction != true label
					successes += (preds != y_val).sum().item()
					total += len(y_val)
				else:
					# Targeted attack success rate: prediction = target label
					mask = y_val != target_label
					successes += (preds[mask] == target_label).sum().item()
					total += mask.sum().item()

		attack_success_rate = successes / total
		prog_bar.set_postfix(f'val_attack_success_rate={attack_success_rate:.4f}')

		if early_stopping(attack_success_rate):  # Aim to maximise validation ASR
			prog_bar.finish()
			early_stopping.print_stop_message()
			break

	early_stopping.restore_best_weights()

	return patch_logits


if __name__ == '__main__':
	# Load data

	train_loader, val_loader, _ = create_data_loaders()

	# Freeze the classifier (only train patches)

	model.eval()
	for param in model.parameters():
		param.requires_grad_(False)

	# Train patches (untargeted then targeted) for each size

	for c in ['untargeted'] + CLASS_SUBSET:
		for patch_size in PATCH_SIZES:
			path = f'./artefacts/patch_{c}_{patch_size}x{patch_size}.pth'
			if Path(path).exists():
				print(path, 'already exists')
			else:
				patch_logits = train_patch(
					patch_size,
					target_class_name=None if c == 'untargeted' else clean_label(c)
				)
				torch.save(patch_logits.detach().cpu(), path)
