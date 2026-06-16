"""
Adversarial perturbation generation

Author: Sam Barba
Created 2026-06-17
"""


def train_noise(target_class_name=None):
	if target_class_name:
		print(f"\nTraining noise (target class = '{target_class_name}')\n")
	else:
		print(f'\nTraining noise (untargeted)\n')

	torch.manual_seed(1)
	torch.cuda.manual_seed_all(1)

	noise_logits = torch.nn.Parameter(0.01 * torch.randn(3, img_size, img_size, device=DEVICE))
	target_label = imagenet_labels.index(target_class_name) if target_class_name else None
	loss_func = torch.nn.CrossEntropyLoss()
	optimiser = torch.optim.Adam([noise_logits], lr=LEARNING_RATE)
	early_stopping = EarlyStopping(target=noise_logits, patience=10, mode='max')

	prog_bar = ProgressBar(range(NUM_EPOCHS), desc='Training', unit='epoch')

	for _ in prog_bar:
		for x_train, y_train in train_loader:
			x_train = x_train.to(DEVICE)
			y_train = y_train.to(DEVICE)

			perturbed = apply_noise(x_train, noise_logits)
			logits = model(perturbed)

			if target_label is None:
				# Untargeted noise objective: maximise loss on the true label (i.e. negate cross-entropy loss)
				loss = -loss_func(logits, y_train)
			else:
				# Targeted noise objective: minimise loss on the target label
				target = torch.full_like(y_train, target_label)
				loss = loss_func(logits, target)

			optimiser.zero_grad()
			loss.backward()
			optimiser.step()

		successes = total = 0

		with torch.inference_mode():
			for x_val, y_val in val_loader:
				perturbed = apply_noise(x_val.to(DEVICE), noise_logits)
				logits = model(perturbed).cpu()
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

	return noise_logits


if __name__ == '__main__':
	print()
