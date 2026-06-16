"""
Utils for Adversarial Attack demo

Author: Sam Barba
Created 2026-06-17
"""

from collections import defaultdict
from pathlib import Path
import random
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from _utils.custom_dataset import CustomDataset
from _utils.progress_bar import ProgressBar


random.seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# For targeted adversarial patches/perturbations, use this subset of ImageNet classes
CLASS_SUBSET = [
	'acoustic_guitar', 'analog_clock', 'espresso', 'giant_panda', 'goldfish', 'koala',
	'lemon', 'starfish', 'tiger', 'toaster', 'violin'
]
SAMPLES_PER_CLASS = 4
PATCH_SIZES = [32, 48, 64]
EPSILON = 0.04
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def clean_label(lbl):
	return re.sub(r"[_'-]", ' ', lbl).lower()


# Define model (ResNet18 with default ImageNet weights)
resnet18_weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=resnet18_weights).to(DEVICE)

transform = resnet18_weights.transforms()
img_size = transform.crop_size[0]
imagenet_labels = [clean_label(i) for i in resnet18_weights.meta['categories']]
transform_mean = torch.tensor(transform.mean, device=DEVICE).view(3, 1, 1)
transform_std = torch.tensor(transform.std, device=DEVICE).view(3, 1, 1)
valid_perturbation_min = (0 - transform_mean) / transform_std
valid_perturbation_max = (1 - transform_mean) / transform_std
pil_image_transform = transforms.Compose([
	transforms.Lambda(lambda img: img * transform_std.cpu() + transform_mean.cpu()),  # De-standardise
	transforms.ToPILImage()
])


def create_data_loaders(batch_size):
	# Sample a few images from every class
	path_groups_by_class = defaultdict(list)
	for p in Path('C:/Users/sam/Desktop/projects/datasets/imagenet').rglob('*.jpg'):
		clean_class = clean_label(p.parent.name)
		if clean_class in imagenet_labels:
			path_groups_by_class[p.parent.name].append(p)

	img_paths = []
	for paths in path_groups_by_class.values():
		img_paths.extend(random.sample(paths, SAMPLES_PER_CLASS))

	x = [
		transform(Image.open(p).convert('RGB')) for p in
		ProgressBar(img_paths, desc='Preprocessing images', unit='imgs')
	]
	y_labels = [clean_label(p.parent.name) for p in img_paths]
	y = torch.tensor([imagenet_labels.index(name) for name in y_labels]).long()

	# Create train/validation/test sets (ratio 0.8:0.1:0.1)
	indices = torch.arange(len(x))
	train_idx, tmp_idx = train_test_split(indices, train_size=0.8, random_state=1)
	val_idx, test_idx = train_test_split(tmp_idx, train_size=0.5, random_state=1)

	x_train = [x[i] for i in train_idx]
	x_val = [x[i] for i in val_idx]
	x_test = [x[i] for i in test_idx]
	y_train = y[train_idx]
	y_val = y[val_idx]
	y_test = y[test_idx]

	train_dataset = CustomDataset(x_train, y_train)
	val_dataset = CustomDataset(x_val, y_val)
	test_dataset = CustomDataset(x_test, y_test)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size)
	test_loader = DataLoader(test_dataset, batch_size=batch_size)

	return train_loader, val_loader, test_loader


def apply_patch(images, patch_logits, validation=False):
	patch = torch.sigmoid(patch_logits)  # Map to [0,1]
	patch = (patch - transform_mean) / transform_std  # Apply same standardisation as ResNet18 transform
	patch_size = patch.shape[1]
	patched = images.clone()

	if validation:
		# For validation, place patch in the same location (image centre)
		min_coord = (img_size - patch_size) // 2
		patched[:, :, min_coord:min_coord + patch_size, min_coord:min_coord + patch_size] = patch
	else:
		# For training/testing, place patch randomly
		coords = torch.randint(0, img_size - patch_size + 1, size=(images.shape[0], 2))
		for i, (y, x) in enumerate(coords):
			patched[i, :, y:y + patch_size, x:x + patch_size] = patch

	return patched


def apply_perturbation(images, perturbation):
	# Convert the perturbation from pixel space into the model's normalised input space. Unlike absolute pixel values,
	# additive perturbations only need to be scaled by the std (no mean subtraction).
	perturbed = images + perturbation / transform_std

	# Clip to the valid image range (corresponds to [0,1] in pixel space)
	perturbed = torch.clamp(perturbed, valid_perturbation_min, valid_perturbation_max)

	return perturbed


def plot_learned_patches(patch_names):
	fig, axes = plt.subplots(nrows=len(PATCH_SIZES), ncols=len(patch_names), figsize=(10, 5))
	plt.subplots_adjust(left=0.08, right=0.96, top=0.82, bottom=0.05, hspace=0.1, wspace=0)
	y_linspace = torch.linspace(0.7, 0.17, len(PATCH_SIZES))
	for s, y in zip(PATCH_SIZES, y_linspace):
		fig.text(x=0.08, y=y, s=f'{s}x{s}', ha='right', va='center', fontsize=11)

	for col, name in enumerate(patch_names):
		axes[0, col].set_title(clean_label(name).capitalize(), fontsize=11)

		for row, size in enumerate(PATCH_SIZES):
			patch_logits = torch.load(f'./artefacts/patch_{name}_{size}x{size}.pth')
			patch = torch.sigmoid(patch_logits)
			patch = patch.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
			axes[row, col].imshow(patch)
			axes[row, col].axis('off')

	plt.suptitle('Learned patches', x=0.52, y=0.95)
	plt.show()


def evaluate(clf_model, test_loader, patch_logits=None, perturbation=None, target_class_name=None, k=5):
	"""
	Evaluate the ResNet18 classifier, optionally applying targeted or untargeted adversarial patches or noise to inputs.
	Computes the classifier's top-1 accuracy and top-k accuracy. If testing with an adversarial patch or perturbation,
	also computes the attack success rate.

	Args:
		clf_model: model to evaluate
		test_loader: test set DataLoader containing images and labels
		patch_logits: optional patch logits to apply to images
		perturbation: optional noise to apply to images
		target_class_name: if patch or perturbation are targeted, this specifies the target class
		k: number of highest-confidence predictions to consider when computing top-k accuracy (default: 5).
	"""

	assert patch_logits is None or perturbation is None, 'Can only test one attack type at a time'

	target_label = imagenet_labels.index(target_class_name) if target_class_name else None

	top1_correct = topk_correct = total = 0  # To calculate top-1 and top-k accuracy
	attack_successes = attack_total = 0  # To calculate Attack Success Rate

	with torch.inference_mode():
		for x, y in test_loader:
			x = x.to(DEVICE)

			if patch_logits is not None:
				x = apply_patch(x, patch_logits)
			elif perturbation is not None:
				x = apply_perturbation(x, perturbation)

			logits = clf_model(x)
			probs = torch.softmax(logits, dim=1).cpu()
			preds1 = probs.argmax(dim=1)
			predsk = torch.topk(probs, k=k, dim=1).indices

			# Update top-1 and top-k accuracy
			top1_correct += (preds1 == y).sum().item()
			topk_correct += predsk.eq(y.view(-1, 1)).any(dim=1).sum().item()
			total += len(y)

			if patch_logits is not None or perturbation is not None:
				# Update ASR
				if target_label is None:
					# Untargeted attack success rate: prediction != true label
					attack_successes += (preds1 != y).sum().item()
					attack_total += len(y)
				else:
					# Targeted attack success rate: prediction = target label
					mask = y != target_label
					attack_successes += (preds1[mask] == target_label).sum().item()
					attack_total += mask.sum().item()

	top1_acc = top1_correct / total
	topk_acc = topk_correct / total

	if attack_total:
		attack_success_rate = attack_successes / attack_total
		return attack_success_rate, top1_acc, topk_acc

	return top1_acc, topk_acc


def plot_output(clf_model, test_batch, patch_logits=None, perturbation=None, target_class_name=None, k=5, title_append=''):
	"""
	Plot the ResNet18 classifier output (top ``k`` predictions for each input), optionally applying targeted or
	untargeted adversarial patches or noise to inputs.

	Args:
		clf_model: model whose outputs to plot
		test_batch: batch of images and labels from test DataLoader
		patch_logits: optional patch logits to apply to images
		perturbation: optional noise to apply to images
		target_class_name: if patch or perturbation are targeted, this specifies the target class
		k: number of highest-confidence predictions to use when plotting model output (default: 5)
		title_append: stats computed in evaluate() to append to title.
	"""

	assert patch_logits is None or perturbation is None, 'Can only test one attack type at a time'

	plt.close()

	x, y = test_batch
	num_samples = len(x)
	target_label = imagenet_labels.index(target_class_name) if target_class_name else None
	samples = []

	with torch.inference_mode():
		x = x.to(DEVICE)

		if patch_logits is not None:
			adversarial = apply_patch(x, patch_logits)
			logits = clf_model(adversarial)
		elif perturbation is not None:
			adversarial = apply_perturbation(x, perturbation)
			logits = clf_model(adversarial)
		else:
			adversarial = torch.empty(len(x))
			logits = clf_model(x)

		probs = torch.softmax(logits, dim=1)

		for img_clean, img_adversarial, true_label, p in zip(x, adversarial, y, probs):
			class_idx = true_label.item()

			# For targeted attacks, don't display images whose class is already the target
			if target_label is not None and class_idx == target_label:
				continue

			samples.append({
				'img_clean': img_clean.clone().cpu(),
				'img_adversarial': img_adversarial.clone().cpu(),
				'true': class_idx,
				'probs': p.clone().cpu()
			})

	# For each image, plot the top 'k' predicted classes

	patch = None

	if patch_logits is None and perturbation is None:
		# Image | bar chart
		fig, axes = plt.subplots(nrows=num_samples, ncols=2, figsize=(8, 7))
		plt.subplots_adjust(left=0.2, right=0.74, top=0.86, bottom=0.05, hspace=0.15, wspace=0)
	else:
		# Image | patch/perturbation | adversarial | bar chart
		fig = plt.figure(figsize=(11, 7))
		outer = fig.add_gridspec(ncols=2, width_ratios=[1.8, 1], left=0.18, right=0.8, top=0.86, bottom=0.05, wspace=0.1)
		left = outer[0].subgridspec(num_samples, 3, wspace=0.1, hspace=0.15)
		right = outer[1].subgridspec(num_samples, 1, hspace=0.15)
		axes = np.empty((num_samples, 4), dtype=object)
		for row in range(num_samples):
			axes[row, 0] = fig.add_subplot(left[row, 0])
			axes[row, 1] = fig.add_subplot(left[row, 1])
			axes[row, 2] = fig.add_subplot(left[row, 2])
			axes[row, 3] = fig.add_subplot(right[row, 0])

		if patch_logits is not None:
			patch = torch.sigmoid(patch_logits.cpu())
			patch = patch.permute(1, 2, 0)
		else:
			perturbation = (perturbation.cpu() + EPSILON) / (2 * EPSILON)  # Map to [0,1] for visualising
			perturbation = perturbation.permute(1, 2, 0)

	text_x = 0.24 if patch is None and perturbation is None else 0.17
	y_linspace = torch.linspace(0.77, 0.14, len(samples))

	for row, example in enumerate(samples):
		# Plot clean image
		ax_img = axes[row, 0]
		ax_img.imshow(pil_image_transform(example['img_clean']))
		ax_img.axis('off')
		fig.text(
			x=text_x, y=y_linspace[row], s=imagenet_labels[example['true']].capitalize(),
			ha='right', va='center', fontsize=10
		)

		if patch is not None or perturbation is not None:
			# Plot patch/perturbation
			ax_patch_perturbation = axes[row, 1]
			ax_patch_perturbation.imshow(patch if patch is not None else perturbation)
			ax_patch_perturbation.axis('off')

			# Plot adversarial image
			ax_adversarial_img = axes[row, 2]
			ax_adversarial_img.imshow(pil_image_transform(example['img_adversarial']))
			ax_adversarial_img.axis('off')

		# Plot top probs
		ax_bar = axes[row, -1]
		top_probs, top_indices = torch.topk(example['probs'], k=k)
		sorted_classes = [imagenet_labels[i.item()] for i in top_indices]
		colours = [
			'tab:green' if idx == example['true']
			else 'tab:red' if idx == target_label
			else 'tab:blue'
			for idx in top_indices
		]
		ax_bar.barh(sorted_classes, top_probs, color=colours)
		ax_bar.invert_yaxis()  # Highest probability at the top
		ax_bar.set_xlim(0, 1)
		ax_bar.tick_params(axis='y', labelsize=9, left=False, right=True, labelleft=False, labelright=True)

		if row < num_samples - 1:
			ax_bar.tick_params(labelbottom=False)

	axes[0, 0].set_title('Clean image', fontsize=11)
	if patch is not None or perturbation is not None:
		axes[0, 1].set_title('Patch' if patch is not None else f'Noise (×{EPSILON})', fontsize=11)
		axes[0, 2].set_title('Adversarial', fontsize=11)
	axes[0, -1].set_title(f'Top {k} softmax probabilities', fontsize=11)

	title = ''
	if patch is None and perturbation is None:
		title = 'No adversarial attack'
	elif patch is None:
		title = f"Perturbation attack (target class = '{target_class_name}')" \
			if target_class_name else 'Perturbation attack (untargeted)'
	elif perturbation is None:
		s = patch.shape[1]
		title = f"{s}x{s} patch attack (target class = '{target_class_name}')" \
			if target_class_name else f'{s}x{s} patch attack (untargeted)'

	plt.suptitle(title + title_append)
	plt.show()
