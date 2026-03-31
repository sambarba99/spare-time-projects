"""
PyTorch classification of patient drawings for Parkinson's disease detection

Author: Sam Barba
Created 27/01/2024
"""

from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from _utils.plotting import *
from conv_net import CNN


np.random.seed(1)
pd.set_option('display.width', None)
pd.set_option('max_colwidth', None)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

IMG_SIZE = 64
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
NUM_EPOCHS = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_data_loaders(df, num_training_augments=11):
	def preprocess_img(path):
		img = cv.imread(path)
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		img = cv.medianBlur(img, 3)  # Denoise
		_, img = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)

		# Make img aspect ratio 1:1
		img_h, img_w = img.shape[:2]

		if img_h > img_w:
			# Extend width of image with black background
			new_w = img_h
			new_img = np.zeros((img_h, new_w), dtype=np.uint8)
			horiz_paste_pos = (new_w - img_w) // 2
			new_img[:, horiz_paste_pos:horiz_paste_pos + img_w] = img
		elif img_h < img_w:
			# Extend height of image with black background
			new_h = img_w
			new_img = np.zeros((new_h, img_w), dtype=np.uint8)
			vert_paste_pos = (new_h - img_h) // 2
			new_img[vert_paste_pos:vert_paste_pos + img_h] = img
		else:
			new_img = img.copy()

		# Dilate drawing so that resizing doesn't "damage" it later
		kernel = np.ones((4, 4), np.uint8)
		new_img = cv.dilate(new_img, kernel)

		new_img = cv.resize(new_img, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_NEAREST)

		# Add channel dim and scale to [0,1]
		new_img = torch.tensor(new_img).float().unsqueeze(dim=0) / 255

		return new_img


	x = [
		preprocess_img(img_path) for img_path in
		tqdm(df['img_path'], desc='Preprocessing images', unit='imgs', ascii=True)
	]
	y = pd.get_dummies(df['class'], prefix='class', drop_first=True, dtype=int).to_numpy().squeeze()
	x = torch.stack(x)
	y = torch.tensor(y).float()

	# Create train/validation/test sets (ratio 0.5:0.25:0.25)
	x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, train_size=0.5, stratify=y, random_state=1)
	x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, train_size=0.5, stratify=y_tmp, random_state=1)

	augment_transform = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		transforms.RandomAffine(
			degrees=0,
			translate=(0.1, 0.1),
			scale=(0.9, 1.1),
			shear=10
		)
	])
	x_train_augmented = torch.stack([
		augment_transform(xi) for xi in x_train
		for _ in range(num_training_augments)
	])
	y_train_augmented = y_train.repeat_interleave(num_training_augments)

	fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(9, 2.5))
	plt.subplots_adjust(left=0.14, right=0.98, top=0.85, bottom=0.08, hspace=0, wspace=0.1)
	fig.text(x=0.125, y=0.66, s='Original', ha='right', va='center', fontsize=12)
	fig.text(x=0.125, y=0.27, s='Augmented', ha='right', va='center', fontsize=12)

	for i in range(8):
		ax1, ax2 = axes[:, i]
		ax1.imshow(x_train[i].squeeze(), cmap='gray')
		ax2.imshow(x_train_augmented[i * num_training_augments].squeeze(), cmap='gray')
		ax1.axis('off')
		ax2.axis('off')

	plt.suptitle('Processed + augmented training samples', x=0.5, y=0.95)
	plt.show()

	x_train = torch.cat([x_train, x_train_augmented])
	y_train = torch.cat([y_train, y_train_augmented])

	train_set = CustomDataset(x_train, y_train)
	val_set = CustomDataset(x_val, y_val)
	test_set = CustomDataset(x_test, y_test)
	train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(val_set, batch_size=len(x_val))
	test_loader = DataLoader(test_set, batch_size=len(x_test))

	return train_loader, val_loader, test_loader


if __name__ == '__main__':
	# Convert data to dataframe

	data = []
	for p in Path('C:/Users/sam/Desktop/projects/datasets/parkinsons').rglob('*.png'):
		class_name = str(p).split('\\')[-2]
		data.append((str(p), class_name))

	df = pd.DataFrame(data, columns=['img_path', 'class'])

	# Plot some examples

	example_indices = [0, 138, 51, 156]
	_, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 5))
	plt.subplots_adjust(top=0.85, bottom=0, hspace=0.1, wspace=0.1)
	for idx, ax in zip(example_indices, axes.flatten()):
		sample = cv.imread(df['img_path'][idx])
		ax.imshow(sample)
		ax.axis('off')
		ax.set_title(df['class'][idx], fontsize=11)
	plt.suptitle('Raw data samples', x=0.514, y=0.95)
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

	train_loader, val_loader, test_loader = create_data_loaders(df)

	model = CNN().to(DEVICE)
	print(f'\nModel:\n{model}\n')
	plot_torch_model(model, (1, IMG_SIZE, IMG_SIZE), device=DEVICE)

	loss_func = torch.nn.BCEWithLogitsLoss()

	if Path('./model.pth').exists():
		model.load_state_dict(torch.load('./model.pth', map_location=DEVICE))
	else:
		# Train model

		print('----- TRAINING -----\n')

		optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
		early_stopping = EarlyStopping(model=model, patience=500, mode='max')

		for epoch in range(1, NUM_EPOCHS + 1):
			progress_bar = tqdm(range(len(train_loader)), unit='batches', ascii=True)
			model.train()

			for x_train, y_train in train_loader:
				progress_bar.update()
				progress_bar.set_description(f'Epoch {epoch}/{NUM_EPOCHS}')

				y_train_logits = model(x_train.to(DEVICE)).squeeze()
				loss = loss_func(y_train_logits, y_train.to(DEVICE))

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

				progress_bar.set_postfix_str(f'loss={loss.item():.4f}')

			model.eval()
			x_val, y_val = next(iter(val_loader))
			with torch.inference_mode():
				y_val_logits = model(x_val.to(DEVICE)).squeeze().cpu()
			y_val_probs = torch.sigmoid(y_val_logits)
			y_val_pred = y_val_probs.round()
			val_loss = loss_func(y_val_logits, y_val).item()
			val_f1 = f1_score(y_val, y_val_pred)
			progress_bar.set_postfix_str(f'val_loss={val_loss:.4f}, val_F1={val_f1:.4f}')
			progress_bar.close()

			if early_stopping(val_f1):
				break

		early_stopping.restore_best_weights()
		torch.save(model.state_dict(), './model.pth')

	# Plot the model's learned filters, and corresponding feature maps of a sample image

	layer_filters = get_cnn_learned_filters(model)
	for idx, (filters, padding, scale_factor) in enumerate(zip(layer_filters, (15, 10), (20, 15)), start=1):
		rows = idx
		cols = len(filters) // rows
		plot_image_grid(
			filters, rows, cols, padding=padding, scale_factor=scale_factor,
			title=f'Filters of conv layer {idx}/{len(layer_filters)}',
			save_path=f'./images/conv{idx}_filters.png'
		)

	x_val, _ = next(iter(val_loader))
	layer_feature_maps = get_cnn_feature_maps(model, input_img=x_val[0].to(DEVICE))
	for idx, (feature_map, padding) in enumerate(zip(layer_feature_maps, (15, 10)), start=1):
		rows = idx
		cols = len(feature_map) // rows
		plot_image_grid(
			feature_map, rows, cols, padding=padding, scale_factor=idx,
			title=f'Feature map of conv layer {idx}/{len(layer_feature_maps)} of a sample image',
			save_path=f'./images/conv{idx}_feature_map.png'
		)

	# Test model (plot confusion matrix and ROC curve)

	print('\n----- TESTING -----\n')

	model.eval()
	x_test, y_test = next(iter(test_loader))
	with torch.inference_mode():
		y_test_logits = model(x_test.to(DEVICE)).squeeze().cpu()
	y_test_probs = torch.sigmoid(y_test_logits)
	y_test_pred = y_test_probs.round()
	test_loss = loss_func(y_test_logits, y_test)
	print('Test loss:', test_loss.item())

	f1 = f1_score(y_test, y_test_pred)
	plot_confusion_matrix(
		y_test,
		y_test_pred,
		['healthy', 'parkinsons'],
		f'Test confusion matrix\n(F1 score: {f1:.3f})'
	)

	plot_roc_curve(y_test, y_test_logits)
