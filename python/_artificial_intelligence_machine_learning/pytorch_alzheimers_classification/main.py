"""
PyTorch classification of brain scans for Alzheimer's disease detection

Author: Sam Barba
Created 27/01/2024
"""

import glob
import os

from cv2 import imread
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
torch.manual_seed(1)

IMG_H = 208
IMG_W = 176
SCALE_FACTOR = 0.5
INPUT_H = round(IMG_H * SCALE_FACTOR)
INPUT_W = round(IMG_W * SCALE_FACTOR)
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
NUM_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_data_loaders(df):
	# Preprocess images now instead of during training (faster pipeline overall)

	transform = transforms.Compose([
		transforms.Resize((INPUT_H, INPUT_W)),
		transforms.Grayscale(),
		transforms.ToTensor()  # Automatically normalises to [0,1]
	])

	x = [
		transform(Image.open(img_path)) for img_path in
		tqdm(df['img_path'], desc='Preprocessing images', unit='imgs', ascii=True)
	]

	label_encoder = LabelEncoder()
	y = label_encoder.fit_transform(df['class'])
	y = torch.tensor(y).long()

	# Create train/validation/test sets (ratio 0.8:0.1:0.1)
	x_train_val, x_test, y_train_val, y_test = train_test_split(
		x, y, train_size=0.9, stratify=y, random_state=1
	)
	x_train, x_val, y_train, y_val = train_test_split(
		x_train_val, y_train_val, train_size=0.89, stratify=y_train_val, random_state=1
	)

	train_dataset = CustomDataset(x_train, y_train)
	val_dataset = CustomDataset(x_val, y_val)
	test_dataset = CustomDataset(x_test, y_test)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
	val_loader = DataLoader(val_dataset, batch_size=len(x_val))
	test_loader = DataLoader(test_dataset, batch_size=len(x_test))

	return train_loader, val_loader, test_loader


if __name__ == '__main__':
	# Convert data to dataframe

	data = []
	for img_path in glob.iglob('C:/Users/Sam/Desktop/projects/datasets/alzheimers/*/*.jpg'):
		class_name = img_path.split('\\')[1]
		data.append((img_path, class_name))

	df = pd.DataFrame(data, columns=['img_path', 'class'])
	print(f'\nRaw data:\n{df}\n')

	# Plot some examples

	example_indices = [0, 1, 3200, 3201, 5440, 5441, 6336, 6337]
	_, axes = plt.subplots(nrows=2, ncols=4, figsize=(6, 4))
	plt.subplots_adjust(top=0.8)
	for idx, ax in zip(example_indices, axes.flatten()):
		sample = imread(df['img_path'][idx])
		ax.imshow(sample)
		ax.axis('off')
		ax.set_title(df['class'][idx][2:].replace('_', ' '), fontsize=11)
	plt.suptitle('Data samples', x=0.514, y=0.94)
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
	plot_torch_model(model, (1, INPUT_H, INPUT_W), input_device=DEVICE)

	loss_func = torch.nn.CrossEntropyLoss()

	if os.path.exists('./model.pth'):
		model.load_state_dict(torch.load('./model.pth', map_location=DEVICE))
	else:
		# Train model

		print('----- TRAINING -----\n')

		optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
		early_stopping = EarlyStopping(patience=10, min_delta=0, mode='max')

		for epoch in range(1, NUM_EPOCHS + 1):
			progress_bar = tqdm(range(len(train_loader)), unit='batches', ascii=True)
			model.train()

			for x_train, y_train in train_loader:
				progress_bar.update()
				progress_bar.set_description(f'Epoch {epoch}/{NUM_EPOCHS}')

				x_train = x_train.to(DEVICE)
				y_train = y_train.to(DEVICE)

				y_train_logits = model(x_train)
				loss = loss_func(y_train_logits, y_train)

				optimiser.zero_grad()
				loss.backward()
				optimiser.step()

				progress_bar.set_postfix_str(f'loss={loss.item():.4f}')

			model.eval()
			x_val, y_val = next(iter(val_loader))
			with torch.inference_mode():
				y_val_logits = model(x_val.to(DEVICE)).cpu()
			val_loss = loss_func(y_val_logits, y_val).item()
			val_f1 = f1_score(y_val, y_val_logits.argmax(dim=1), average='weighted')
			progress_bar.set_postfix_str(f'val_loss={val_loss:.4f}, val_F1={val_f1:.4f}')
			progress_bar.close()

			if early_stopping(val_f1, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './model.pth')

	# Plot the model's learned filters, and corresponding feature maps of a sample image

	layer_filters = get_cnn_learned_filters(model)
	for idx, (filters, gap, scale_factor) in enumerate(zip(layer_filters, (15, 10), (20, 15)), start=1):
		cols = idx * 8
		rows = len(filters) // cols
		plot_image_grid(
			filters, rows, cols, gap=gap, scale_factor=scale_factor,
			title=f'Filters of conv layer {idx}/{len(layer_filters)}',
			save_path=f'./images/conv{idx}_filters.png'
		)

	x_val, _ = next(iter(val_loader))
	layer_feature_maps = get_cnn_feature_maps(model, input_img=x_val[0].to(DEVICE))
	for idx, (feature_map, gap, scale_factor) in enumerate(zip(layer_feature_maps, (10, 5), (0.75, 1)), start=1):
		cols = idx * 8
		rows = len(feature_map) // cols
		plot_image_grid(
			feature_map, rows, cols, gap=gap, scale_factor=scale_factor,
			title=f'Feature map of conv layer {idx}/{len(layer_feature_maps)} of a sample image',
			save_path=f'./images/conv{idx}_feature_map.png'
		)

	# Test model

	print('\n----- TESTING -----\n')

	model.eval()
	x_test, y_test = next(iter(test_loader))
	with torch.inference_mode():
		y_test_logits = model(x_test.to(DEVICE)).cpu()

	test_loss = loss_func(y_test_logits, y_test)
	print('Test loss:', test_loss.item())

	# Confusion matrix
	f1 = f1_score(y_test, y_test_logits.argmax(dim=1), average='weighted')
	plot_confusion_matrix(
		y_test,
		y_test_logits.argmax(dim=1),
		['healthy', 'very mild', 'mild', 'moderate'],
		f'Test confusion matrix\n(F1 score: {f1:.3f})',
		x_ticks_rotation=45,
		horiz_alignment='right'
	)
