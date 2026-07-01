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

# For targeted adversarial patches/noise, use this subset of ImageNet classes
CLASS_SUBSET = [
	'acoustic_guitar', 'analog_clock', 'espresso', 'fire_engine', 'giant_panda', 'goldfish', 'koala', 'lemon', 'pizza',
	'starfish', 'tiger', 'toaster', 'violin'
]
SAMPLES_PER_CLASS = 4
BATCH_SIZE = 32
PATCH_SIZES = [32, 48, 64]
EPSILON = 0.02
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
valid_min = (0 - transform_mean) / transform_std
valid_max = (1 - transform_mean) / transform_std
pil_image_transform = transforms.Compose([
	transforms.Lambda(lambda img: img * transform_std.cpu() + transform_mean.cpu()),  # De-standardise
	transforms.ToPILImage()
])


def create_data_loaders():
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
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

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


def apply_noise(images, noise_logits):
	noise = EPSILON * torch.tanh(noise_logits)  # Map to [-EPSILON, EPSILON]

	# Convert the perturbation from pixel space into the model's normalised input space. Unlike absolute pixel values,
	# additive perturbations only need to be scaled by the std (no mean subtraction).
	noise /= transform_std

	perturbed = images + noise

	# Clip to the valid image range (equivalent to [0,1] in pixel space)
	perturbed = torch.clamp(perturbed, valid_min, valid_max)

	return perturbed


def plot_learned_patches(patch_names, patch_sizes):
	fig, axes = plt.subplots(nrows=len(patch_sizes), ncols=len(patch_names), figsize=(12, 5))
	plt.subplots_adjust(left=0.1, right=0.95, top=0.8, bottom=0.05, hspace=0.1, wspace=0.1)
	y_linspace = torch.linspace(0.68, 0.17, len(patch_sizes))
	for s, y in zip(patch_sizes, y_linspace):
		fig.text(x=0.1, y=y, s=f'{s}x{s}', ha='right', va='center', fontsize=11)

	for col, name in enumerate(patch_names):
		axes[0, col].set_title(clean_label(name).capitalize(), fontsize=11)

		for row, size in enumerate(patch_sizes):
			patch_logits = torch.load(f'./artefacts/patch_{name}_{size}x{size}.pth')
			patch = torch.sigmoid(patch_logits)
			patch = patch.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
			axes[row, col].imshow(patch)
			axes[row, col].axis('off')

	plt.suptitle('Learned patches', x=0.525, y=0.95)
	plt.show()


def evaluate(clf_model, loader, patch_logits=None, noise_logits=None, target_class_name=None, num_images=5, k=5):
	"""
	Evaluate the ResNet18 classifier, optionally applying targeted or untargeted adversarial patches or noise to inputs.
	Samples ``num_images`` images from ``loader`` (selecting at most one image per class), and plots the classifier's
	top ``k`` predictions for each.

	Args:
		clf_model: model to evaluate
		loader: DataLoader containing images and labels
		patch_logits: optional patch logits to apply to images
		noise_logits: optional noise logits to apply to images
		target_class_name: if patch or noise are targeted, this specifies the target class
		num_images: number of images to sample for evaluation (one per unique class)
		k: number of top predictions to plot for each input image
	"""

	assert patch_logits is None or noise_logits is None, 'Can only test one attack type at a time'

	target_label = imagenet_labels.index(target_class_name) if target_class_name else None
	seen_classes = set()
	examples = []

	top1_correct = topk_correct = total = 0  # To calculate top-1 and top-k accuracy
	attack_successes = attack_total = 0  # To calculate Attack Success Rate

	with torch.inference_mode():
		for x, y in loader:
			x = x.to(DEVICE)

			if patch_logits is not None:
				adversarial = apply_patch(x, patch_logits)
				logits = clf_model(adversarial)
			elif noise_logits is not None:
				adversarial = apply_noise(x, noise_logits)
				logits = clf_model(adversarial)
			else:
				adversarial = torch.empty(len(x))
				logits = clf_model(x)

			probs = torch.softmax(logits, dim=1).cpu()
			preds1 = probs.argmax(dim=1)
			predsk = torch.topk(probs, k=k, dim=1).indices

			# Update top-1 and top-k accuracy
			top1_correct += (preds1 == y).sum().item()
			topk_correct += predsk.eq(y.view(-1, 1)).any(dim=1).sum().item()
			total += len(y)

			if patch_logits is not None or noise_logits is not None:
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

			if len(examples) < num_images:
				for img_clean, img_adversarial, true_label, p in zip(x, adversarial, y, probs):
					class_idx = true_label.item()

					if class_idx in seen_classes:
						continue

					# For targeted attacks, don't display images whose class is already the target
					if target_label is not None and class_idx == target_label:
						continue

					seen_classes.add(class_idx)
					examples.append({
						'img_clean': img_clean.clone().cpu(),
						'img_adversarial': img_adversarial.clone().cpu(),
						'true': class_idx,
						'probs': p.clone()
					})

					if len(examples) == num_images:
						break

	examples.sort(key=lambda i: imagenet_labels[i['true']])  # Sort alphabetically

	# For each image, plot the top 'k' predicted classes

	patch = noise = None

	if patch_logits is None and noise_logits is None:
		# Image | Bar chart
		_, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(8, 8))
		plt.subplots_adjust(left=0.04, right=0.87, bottom=0.05, hspace=0.2, wspace=0.6)
	else:
		# Image | Patch/Noise | Adversarial     |     Bar chart
		fig = plt.figure(figsize=(10, 8))
		outer = fig.add_gridspec(ncols=2, width_ratios=[1.8, 1], left=0.04, right=0.94, bottom=0.05, wspace=0.5)
		left = outer[0].subgridspec(num_images, 3, wspace=0.01, hspace=0.2)
		right = outer[1].subgridspec(num_images, 1, hspace=0.2)
		axes = np.empty((num_images, 4), dtype=object)
		for row in range(num_images):
			axes[row, 0] = fig.add_subplot(left[row, 0])
			axes[row, 1] = fig.add_subplot(left[row, 1])
			axes[row, 2] = fig.add_subplot(left[row, 2])
			axes[row, 3] = fig.add_subplot(right[row, 0])

		if patch_logits is not None:
			patch = torch.sigmoid(patch_logits.cpu())
			patch = patch.permute(1, 2, 0)
		else:
			noise = EPSILON * torch.tanh(noise_logits.cpu())
			noise = noise.permute(1, 2, 0)
			noise = (noise + EPSILON) / (2 * EPSILON)  # Map to [0,1] to visualise

	for row, example in enumerate(examples):
		# Plot clean image
		ax_img = axes[row, 0]
		ax_img.imshow(pil_image_transform(example['img_clean']))
		ax_img.axis('off')
		ax_img.set_title(imagenet_labels[example['true']].capitalize(), fontsize=10, y=0.97)

		if patch is not None or noise is not None:
			# Plot patch/noise
			ax_patch_noise = axes[row, 1]
			ax_patch_noise.imshow(patch if patch is not None else noise)
			ax_patch_noise.axis('off')

			# Plot adversarial image
			ax_adversarial_img = axes[row, 2]
			ax_adversarial_img.imshow(pil_image_transform(example['img_adversarial']))
			ax_adversarial_img.axis('off')

		# Plot top probs
		ax_bar = axes[row, -1]
		top_probs, top_indices = torch.topk(example['probs'], k=k)
		sorted_classes = [imagenet_labels[i.item()] for i in top_indices]
		colours = [
			'tab:red' if idx == target_label
			else 'tab:blue'
			for idx in top_indices
		]
		ax_bar.barh(sorted_classes, top_probs, color=colours)
		ax_bar.invert_yaxis()  # Highest probability at the top
		ax_bar.set_xlim(0, 1)
		ax_bar.tick_params(axis='y', labelsize=9)

		if row < num_images - 1:
			ax_bar.tick_params(labelbottom=False)

	if patch is not None or noise is not None:
		axes[0, 1].set_title('Patch' if patch is not None else 'Noise', fontsize=10, y=0.97)
		axes[0, 2].set_title('Adversarial', fontsize=10, y=0.97)
	axes[0, -1].set_title(f'Top {k} softmax probabilities', fontsize=10, y=0.97)

	title = ''
	if patch is None and noise is None:
		title = 'Model evaluation (no attack)'
	elif patch is None:
		title = f"Model evaluation (perturbation attack, target class = '{target_class_name}')" \
			if target_class_name else 'Model evaluation (untargeted perturbation attack)'
	elif noise is None:
		s = patch.shape[1]
		title = f"Model evaluation ({s}x{s} patch attack, target class = '{target_class_name}')" \
			if target_class_name else f'Model evaluation (untargeted {s}x{s} patch attack)'

	top1_acc = top1_correct / total
	topk_acc = topk_correct / total
	title += f'\nTop-1 acc: {top1_acc:.3f}  |  Top-{k} acc: {topk_acc:.3f}'

	if attack_total:
		attack_success_rate = attack_successes / attack_total
		title += f'  |  Attack success rate: {attack_success_rate:.3f}'

	plt.suptitle(title)
	plt.show()
