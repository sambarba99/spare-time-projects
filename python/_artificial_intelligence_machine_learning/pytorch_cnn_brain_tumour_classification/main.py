"""
Classification of brain tumour MRIs with a PyTorch Convolutional Neural Network (CNN)

Author: Sam Barba
Created 09/04/2026
"""

from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from _utils.plotting import *
from conv_net import CNN


pd.set_option('display.width', None)
pd.set_option('max_colwidth', None)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

IMG_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def pad_to_square(img):
	w, h = img.size

	if h > w:
		pad_left = (h - w) // 2
		pad_right = h - w - pad_left
		padding = (pad_left, 0, pad_right, 0)
	elif h < w:
		pad_top = (w - h) // 2
		pad_bottom = w - h - pad_top
		padding = (0, pad_top, 0, pad_bottom)
	else:
		return img

	return transforms.functional.pad(img, padding)


def create_data_loaders(df):
	transform = transforms.Compose([
		transforms.Lambda(pad_to_square),
		transforms.Resize(IMG_SIZE),
		transforms.Grayscale(),
		transforms.ToTensor()  # Scale to [0,1]
	])

	x = [
		transform(Image.open(img_path)) for img_path in
		tqdm(df['img_path'], desc='Preprocessing images', unit='imgs', ascii=True)
	]
	x = torch.stack(x)
	label_encoder = LabelEncoder()
	y = label_encoder.fit_transform(df['class'])
	y = torch.tensor(y).long()
	labels = sorted(df['class'].unique())

	# Create train/validation/test sets (ratio 0.7:0.15:0.15)
	x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, train_size=0.7, stratify=y, random_state=1)
	x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, train_size=0.5, stratify=y_tmp, random_state=1)

	augment_transform = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		transforms.RandomAffine(
			degrees=10,
			translate=(0.1, 0.1),
			scale=(0.9, 1.1)
		)
	])
	x_train_augmented = torch.stack([augment_transform(xi) for xi in x_train])
	y_train_augmented = y_train.clone()

	fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(9, 2.5))
	plt.subplots_adjust(left=0.14, right=0.98, top=0.82, bottom=0.05, hspace=0, wspace=0.1)
	fig.text(x=0.125, y=0.63, s='Original', ha='right', va='center', fontsize=12)
	fig.text(x=0.125, y=0.25, s='Augmented', ha='right', va='center', fontsize=12)

	for i in range(8):
		ax1, ax2 = axes[:, i]
		ax1.imshow(x_train[i].squeeze(), cmap='gray')
		ax2.imshow(x_train_augmented[i].squeeze(), cmap='gray')
		ax1.axis('off')
		ax2.axis('off')

	plt.suptitle('Processed + augmented training samples', x=0.5, y=0.95)
	plt.show()

	x_train = torch.cat([x_train, x_train_augmented])
	y_train = torch.cat([y_train, y_train_augmented])

	# Standardise images using training mean and std
	mean = torch.zeros(1)
	sq_mean = torch.zeros(1)
	num_pixels = IMG_SIZE * IMG_SIZE * len(x_train)
	for img in tqdm(x_train, desc='Calculating mean and std of x_train', unit='imgs', ascii=True):
		mean += img.sum(dim=[1, 2])
		sq_mean += (img ** 2).sum(dim=[1, 2])
	mean /= num_pixels
	sq_mean /= num_pixels
	std = (sq_mean - mean ** 2).sqrt()
	norm_transform = transforms.Normalize(mean.tolist(), std.tolist())
	for idx, img in enumerate(x_train):
		x_train[idx] = norm_transform(img)
	for idx, img in enumerate(x_val):
		x_val[idx] = norm_transform(img)
	for idx, img in enumerate(x_test):
		x_test[idx] = norm_transform(img)

	train_dataset = CustomDataset(x_train, y_train)
	val_dataset = CustomDataset(x_val, y_val)
	test_dataset = CustomDataset(x_test, y_test)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

	return train_loader, val_loader, test_loader, labels


if __name__ == '__main__':
	# Convert data to dataframe

	data = []
	for p in Path('C:/Users/sam/Desktop/projects/datasets/brain_tumours').rglob('*.jpg'):
		y = str(p).split('\\')[-2]
		if y == 'notumor':
			y = 'no tumour'
		data.append((str(p), y))

	df = pd.DataFrame(data, columns=['img_path', 'class'])

	print(f'\nRaw data:\n{df}\n')

	# Plot some examples

	first_2_indices_per_class = df.groupby('class', sort=False).head(2).index
	_, axes = plt.subplots(nrows=2, ncols=4, figsize=(6.5, 4))
	plt.subplots_adjust(top=0.8)
	for idx, ax in zip(first_2_indices_per_class, axes.flatten()):
		sample = cv.imread(df['img_path'][idx])
		ax.imshow(sample)
		ax.axis('off')
		ax.set_title(df['class'][idx], fontsize=11)
	plt.suptitle('Raw data samples', x=0.514, y=0.94)
	plt.gcf().set_facecolor('#80b0f0')
	plt.show()

	# Plot output feature (class) distributions

	unique_value_counts = df['class'].value_counts()
	plt.bar(unique_value_counts.index, unique_value_counts.values)
	plt.xlabel('Class')
	plt.ylabel('Count')
	plt.title('Class distribution')
	plt.show()

	# Define data loaders and model

	train_loader, val_loader, test_loader, labels = create_data_loaders(df)

	model = CNN().to(DEVICE)
	print(f'\nModel:\n{model}\n')
	plot_torch_model(model, (1, IMG_SIZE, IMG_SIZE), device=DEVICE)

	loss_func = torch.nn.CrossEntropyLoss()

	if Path('./model.pth').exists():
		model.load_state_dict(torch.load('./model.pth', map_location=DEVICE))
	else:
		# Train model

		print('----- TRAINING -----\n')

		optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
		early_stopping = EarlyStopping(model=model, patience=10, mode='max')

		for epoch in range(1, NUM_EPOCHS + 1):
			progress_bar = tqdm(range(len(train_loader)), unit='batches', ascii=True)
			model.train()

			for x_train, y_train in train_loader:
				progress_bar.update()
				progress_bar.set_description(f'Epoch {epoch}/{NUM_EPOCHS}')

				y_train_logits = model(x_train.to(DEVICE))
				loss = loss_func(y_train_logits, y_train.to(DEVICE))

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

				progress_bar.set_postfix_str(f'loss={loss.item():.4f}')

			val_loss_total = 0
			all_y_labels, all_y_preds = [], []

			model.eval()
			with torch.inference_mode():
				for x_val, y_val in val_loader:
					y_val_logits = model(x_val.to(DEVICE)).cpu()

					val_loss_total += loss_func(y_val_logits, y_val).item()
					all_y_labels.append(y_val)
					all_y_preds.append(y_val_logits.argmax(dim=1))

			val_loss = val_loss_total / len(val_loader)
			val_f1 = f1_score(torch.cat(all_y_labels), torch.cat(all_y_preds), average='weighted')
			progress_bar.set_postfix_str(f'val_loss={val_loss:.4f}, val_F1={val_f1:.4f}')
			progress_bar.close()

			if early_stopping(val_f1):
				break

		early_stopping.restore_best_weights()
		torch.save(model.state_dict(), './model.pth')

	# Plot the model's learned filters, and corresponding feature maps of a sample image

	layer_filters = get_cnn_learned_filters(model)
	for idx, (filters, rows, scale_factor) in enumerate(zip(layer_filters, (1, 2, 4), (15, 20, 20)), start=1):
		cols = len(filters) // rows
		plot_image_grid(
			filters, rows, cols, padding=15, scale_factor=scale_factor,
			title=f'Filters of conv layer {idx}/{len(layer_filters)}',
			save_path=f'./images/conv{idx}_filters.png'
		)

	x_val, _ = next(iter(val_loader))
	layer_feature_maps = get_cnn_feature_maps(model, input_img=x_val[0].to(DEVICE))
	for idx, (feature_map, rows, scale_factor) in enumerate(zip(layer_feature_maps, (1, 2, 4), (0.5, 1, 2)), start=1):
		cols = len(feature_map) // rows
		plot_image_grid(
			feature_map, rows, cols, padding=10, scale_factor=scale_factor,
			title=f'Feature map of conv layer {idx}/{len(layer_feature_maps)} of a sample image',
			save_path=f'./images/conv{idx}_feature_map.png'
		)

	# Test model

	print('\n----- TESTING -----\n')

	test_loss_total = 0
	all_y_labels, all_y_preds = [], []

	model.eval()
	with torch.inference_mode():
		for x_test, y_test in test_loader:
			y_test_logits = model(x_test.to(DEVICE)).cpu()

			test_loss_total += loss_func(y_test_logits, y_test).item()
			all_y_labels.append(y_test)
			all_y_preds.append(y_test_logits.argmax(dim=1))

	test_loss = test_loss_total / len(test_loader)
	print('Test loss:', test_loss)

	# Confusion matrix
	f1 = f1_score(torch.cat(all_y_labels), torch.cat(all_y_preds), average='weighted')
	plot_confusion_matrix(
		torch.cat(all_y_labels),
		torch.cat(all_y_preds),
		labels,
		f'Test confusion matrix\n(F1 score: {f1:.3f})',
		x_ticks_rotation=45,
		horiz_alignment='right'
	)
