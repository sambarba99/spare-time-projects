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
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from _utils.model_architecture_plots import plot_model
from _utils.model_evaluation_plots import plot_cnn_learned_filters, plot_cnn_feature_maps, plot_confusion_matrix
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

	y = pd.get_dummies(df['class'], prefix='class', dtype=int).to_numpy().squeeze()
	y = torch.tensor(y).float()

	# Create train/validation/test sets (ratio 0.8:0.1:0.1)
	x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, train_size=0.9, stratify=y, random_state=1)
	x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, train_size=0.89, stratify=y_train_val, random_state=1)

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
	_, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 6))
	plt.subplots_adjust(top=0.85, bottom=0.05, hspace=0, wspace=0.05)
	for idx, ax in zip(example_indices, axes.flatten()):
		sample = imread(df['img_path'][idx])
		ax.imshow(sample)
		ax.axis('off')
		ax.set_title(df['class'][idx][2:].replace('_', ' '))
	plt.suptitle('Data samples', x=0.514, y=0.94)
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

	model = CNN()
	print(f'\nModel:\n{model}\n')
	plot_model(model, (1, INPUT_H, INPUT_W))
	model.to(DEVICE)

	loss_func = torch.nn.CrossEntropyLoss()

	if os.path.exists('./model.pth'):
		model.load_state_dict(torch.load('./model.pth'))
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
			x_val = x_val.to(DEVICE)
			with torch.inference_mode():
				y_val_logits = model(x_val).cpu()
			val_loss = loss_func(y_val_logits, y_val).item()
			val_f1 = f1_score(y_val.argmax(dim=1), y_val_logits.argmax(dim=1), average='weighted')
			progress_bar.set_postfix_str(f'val_loss={val_loss:.4f}, val_F1={val_f1:.4f}')
			progress_bar.close()

			if early_stopping(val_f1, model.state_dict()):
				print('Early stopping at epoch', epoch)
				break

		model.load_state_dict(early_stopping.best_weights)  # Restore best weights
		torch.save(model.state_dict(), './model.pth')

	# Plot the model's learned filters, and corresponding feature maps of a sample image

	plot_cnn_learned_filters(model, num_cols=8)

	x_val, _ = next(iter(val_loader))
	plot_cnn_feature_maps(model, num_cols=8, input_img=x_val[0].to(DEVICE), title_append=' of a sample image')

	# Test model

	print('\n----- TESTING -----\n')

	model.eval()
	x_test, y_test = next(iter(test_loader))
	x_test = x_test.to(DEVICE)
	with torch.inference_mode():
		y_test_logits = model(x_test).cpu()

	test_loss = loss_func(y_test_logits, y_test)
	print('Test loss:', test_loss.item())

	# Confusion matrix
	f1 = f1_score(y_test.argmax(dim=1), y_test_logits.argmax(dim=1), average='weighted')
	plot_confusion_matrix(
		y_test.argmax(dim=1),
		y_test_logits.argmax(dim=1),
		['healthy', 'very mild', 'mild', 'moderate'],
		f'Test confusion matrix\n(F1 score: {f1:.3f})',
		x_ticks_rotation=45,
		horiz_alignment='right'
	)
