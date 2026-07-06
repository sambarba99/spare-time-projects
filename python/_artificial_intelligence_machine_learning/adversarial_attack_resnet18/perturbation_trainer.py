"""
Universal Adversarial Perturbation training (targeted + untargeted)

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
torch.use_deterministic_algorithms(True)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

BATCH_SIZE = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 200


def train_perturbation(target_class_name=None):
	if target_class_name:
		print(f"\nTraining perturbation (target class = '{target_class_name}')\n")
	else:
		print(f'\nTraining perturbation (untargeted)\n')

	torch.manual_seed(1)
	torch.cuda.manual_seed_all(1)

	perturbation = torch.nn.Parameter(0.01 * torch.randn(3, img_size, img_size, device=DEVICE))
	target_label = imagenet_labels.index(target_class_name) if target_class_name else None
	loss_func = torch.nn.CrossEntropyLoss()
	optimiser = torch.optim.Adam([perturbation], lr=LEARNING_RATE)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='max', factor=0.5, patience=10, min_lr=1e-3)
	early_stopping = EarlyStopping(target=perturbation, patience=20, mode='max')

	prog_bar = ProgressBar(range(NUM_EPOCHS), unit='epoch')

	for _ in prog_bar:
		for x_train, y_train in train_loader:
			x_train = x_train.to(DEVICE)
			y_train = y_train.to(DEVICE)

			perturbed = apply_perturbation(x_train, perturbation)
			logits = model(perturbed)

			if target_label is None:
				# Untargeted perturbation objective: maximise loss on the true label (i.e. negate cross-entropy loss)
				loss = -loss_func(logits, y_train)
			else:
				# Targeted perturbation objective: minimise loss on the target label
				target = torch.full_like(y_train, target_label)
				loss = loss_func(logits, target)

			optimiser.zero_grad()
			loss.backward()
			optimiser.step()

			with torch.no_grad():
				perturbation.clamp_(-EPSILON, EPSILON)

		successes = total = 0

		with torch.inference_mode():
			for x_val, y_val in val_loader:
				perturbed = apply_perturbation(x_val.to(DEVICE), perturbation)
				logits = model(perturbed).cpu()
				preds = logits.argmax(dim=1)

				if target_label is None:
					# Untargeted attack success rate: prediction != true label
					successes += (preds != y_val).sum().item()
					total += y_val.shape[0]
				else:
					# Targeted attack success rate: prediction = target label
					mask = y_val != target_label
					successes += (preds[mask] == target_label).sum().item()
					total += mask.sum().item()

		val_attack_success_rate = successes / total
		prog_bar.set_postfix(f"{val_attack_success_rate=:.4f}, lr={optimiser.param_groups[0]['lr']:.3e}")
		scheduler.step(val_attack_success_rate)

		if early_stopping(val_attack_success_rate):  # Aim to maximise validation ASR
			prog_bar.finish()
			early_stopping.print_stop_message()
			break

	early_stopping.restore_best_weights()

	return perturbation


if __name__ == '__main__':
	# Load data

	train_loader, val_loader, _ = create_data_loaders(BATCH_SIZE)

	# Freeze the classifier (only train perturbations)

	model.eval()
	for param in model.parameters():
		param.requires_grad_(False)

	# Train perturbations (untargeted then targeted)

	for c in ['untargeted'] + CLASS_SUBSET:
		path = f'./artefacts/perturbation_{c}.pth'
		if Path(path).exists():
			print(f'\n{path} already exists')
		else:
			perturbation = train_perturbation(target_class_name=None if c == 'untargeted' else clean_label(c))
			torch.save(perturbation.detach().cpu(), path)
