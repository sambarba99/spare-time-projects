"""
Knowledge Distillation demo using a subset of the CIFAR-10 dataset

Author: Sam Barba
Created 29/09/2024
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from _utils.custom_dataset import CustomDataset
from _utils.early_stopping import EarlyStopping
from _utils.model_plotting import plot_torch_model, plot_confusion_matrix
from conv_nets import Teacher, Student


torch.manual_seed(1)

IMG_SIZE = 32
BATCH_SIZE = 128
NUM_EPOCHS = 100
DISTILLATION_LOSS_WEIGHT = 0.6   # Contribution of distillation loss to KD training
CROSS_ENTROPY_LOSS_WEIGHT = 0.4  # Contribution of cross-entropy loss to KD training
TEMPERATURE = 2                  # Controls smoothness of output distributions
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_data_loaders():
	# Preprocess images now instead of during training (faster pipeline overall)

	transform = transforms.ToTensor()  # Normalise to [0,1]

	img_paths = glob.glob('C:/Users/Sam/Desktop/projects/datasets/cifar10/*.png')
	x = [
		transform(Image.open(img_path)) for img_path in
		tqdm(img_paths, desc='Preprocessing images', unit='imgs', ascii=True)
	]
	y_labels = [img_path.split('\\')[-1].split('_')[0] for img_path in img_paths]

	# One-hot encode y
	one_hot_encoder = OneHotEncoder(sparse_output=False)
	y = np.array(y_labels).reshape(-1, 1)
	y = one_hot_encoder.fit_transform(y)

	# Create train/validation/test sets (ratio 0.95:0.04:0.01)

	indices = np.arange(len(x))
	train_val_idx, test_idx = train_test_split(indices, train_size=0.99, stratify=y, random_state=1)
	train_idx, val_idx = train_test_split(train_val_idx, train_size=0.96, stratify=y[train_val_idx], random_state=1)

	x_train = [x[i] for i in train_idx]
	x_val = [x[i] for i in val_idx]
	x_test = [x[i] for i in test_idx]
	y_train = y[train_idx]
	y_val = y[val_idx]
	y_test = y[test_idx]
	test_labels = [y_labels[i] for i in test_idx]

	train_dataset = CustomDataset(x_train, y_train)
	val_dataset = CustomDataset(x_val, y_val)
	test_dataset = CustomDataset(x_test, y_test, test_labels)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
	val_loader = DataLoader(val_dataset, batch_size=len(x_val))
	test_loader = DataLoader(test_dataset, batch_size=len(x_test))

	return train_loader, val_loader, test_loader


def train(model, save_path):
	torch.manual_seed(1)  # Ensure equal training for all models

	loss_func = torch.nn.CrossEntropyLoss()
	optimiser = torch.optim.Adam(model.parameters())  # LR = 1e-3
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
		val_f1 = f1_score(y_val.argmax(dim=1), y_val_logits.argmax(dim=1), average='weighted')
		progress_bar.set_postfix_str(f'val_loss={val_loss:.4f}, val_F1={val_f1:.4f}')
		progress_bar.close()

		if early_stopping(val_f1, model.state_dict()):
			print('Early stopping at epoch', epoch)
			break

	model.load_state_dict(early_stopping.best_weights)  # Restore best weights
	torch.save(model.state_dict(), save_path)


def train_student_with_kd(teacher_model, student_model, save_path):
	torch.manual_seed(1)

	loss_func = torch.nn.CrossEntropyLoss()
	optimiser = torch.optim.Adam(student_model.parameters())
	early_stopping = EarlyStopping(patience=10, min_delta=0, mode='max')

	teacher_model.eval()

	for epoch in range(1, NUM_EPOCHS + 1):
		progress_bar = tqdm(range(len(train_loader)), unit='batches', ascii=True)
		student_model.train()

		for x_train, y_train in train_loader:
			progress_bar.update()
			progress_bar.set_description(f'Epoch {epoch}/{NUM_EPOCHS}')

			x_train = x_train.to(DEVICE)
			y_train = y_train.to(DEVICE)

			with torch.inference_mode():
				teacher_logits = teacher_model(x_train)
			student_logits = student_model(x_train)

			teacher_soft_probs = torch.softmax(teacher_logits / TEMPERATURE, dim=-1)
			student_soft_probs = torch.log_softmax(student_logits / TEMPERATURE, dim=-1)

			# Calculate the distillation loss (using KL divergence),
			# scaled by T^2 (source: https://arxiv.org/pdf/1503.02531)
			distillation_loss = torch.nn.functional.kl_div(
				student_soft_probs,
				teacher_soft_probs,
				reduction='batchmean'
			) * TEMPERATURE * TEMPERATURE

			# Calculate the true label loss
			label_loss = loss_func(student_logits, y_train)

			# Weighted sum of the losses
			loss = DISTILLATION_LOSS_WEIGHT * distillation_loss + CROSS_ENTROPY_LOSS_WEIGHT * label_loss

			optimiser.zero_grad()
			loss.backward()
			optimiser.step()

			progress_bar.set_postfix_str(f'loss={loss.item():.4f}')

		student_model.eval()
		x_val, y_val = next(iter(val_loader))
		with torch.inference_mode():
			y_val_logits = student_model(x_val.to(DEVICE)).cpu()

		val_loss = loss_func(y_val_logits, y_val).item()
		val_f1 = f1_score(y_val.argmax(dim=1), y_val_logits.argmax(dim=1), average='weighted')
		progress_bar.set_postfix_str(f'val_loss={val_loss:.4f}, val_F1={val_f1:.4f}')
		progress_bar.close()

		if early_stopping(val_f1, student_model.state_dict()):
			print('Early stopping at epoch', epoch)
			break

	student_model.load_state_dict(early_stopping.best_weights)  # Restore best weights
	torch.save(student_model.state_dict(), save_path)


def test(model, plot_title):
	model.eval()
	x_test, y_test, y_labels = next(iter(test_loader))
	with torch.inference_mode():
		y_test_logits = model(x_test.to(DEVICE)).cpu()

	ordered_y_labels = sorted(set(y_labels))
	y_test = y_test.argmax(dim=1)
	y_test_pred = y_test_logits.argmax(dim=1)

	# Confusion matrix
	f1 = f1_score(y_test, y_test_pred, average='weighted')
	print(f'F1 score: {f1:.3f}')
	plot_confusion_matrix(
		y_test,
		y_test_pred,
		ordered_y_labels,
		f'Test confusion matrix\n(F1 score: {f1:.3f})',
		x_ticks_rotation=45,
		horiz_alignment='right'
	)

	# Convert logits to probs
	y_test_probs = torch.softmax(y_test_logits, dim=-1)

	# Find highest probs (the predicted class certainties)
	y_test_probs, _ = y_test_probs.max(dim=1)

	# Convert to percentages
	y_test_probs *= 100

	# Plot 16 test set images with outputs (this list contains at least 1 of every class)

	test_indices = [
		36, 4, 0, 18, 45, 7, 28, 15,
		12, 1, 56, 8, 5, 25, 52, 10
	]

	_, axes = plt.subplots(nrows=4, ncols=4, figsize=(9, 6))
	plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05, hspace=0.45)

	pil_image_transform = transforms.ToPILImage()

	for idx, ax in zip(test_indices, axes.flatten()):
		img, y_pred_class_idx, y_pred_prob, y_label = x_test[idx], y_test_pred[idx], y_test_probs[idx], y_labels[idx]

		ax.imshow(pil_image_transform(img))
		ax.axis('off')
		ax.set_title(
			f'Pred: {ordered_y_labels[y_pred_class_idx]} ({y_pred_prob:.1f}%)'
			f'\nActual: {y_label}',
			fontsize=10,
			y=0.96
		)

	plt.suptitle(plot_title, y=0.96)
	plt.show()


if __name__ == '__main__':
	# Load data

	train_loader, val_loader, test_loader = create_data_loaders()

	# Define models

	teacher_model = Teacher().to(DEVICE)
	student_model_no_kd = Student().to(DEVICE)
	student_model_with_kd = Student().to(DEVICE)
	student_model_with_kd.load_state_dict(student_model_no_kd.state_dict())
	print(f'\nTeacher model:\n{teacher_model}')
	print(f'\nStudent model:\n{student_model_no_kd}')
	plot_torch_model(teacher_model, (3, IMG_SIZE, IMG_SIZE), input_device=DEVICE, out_file='./images/teacher_architecture')
	plot_torch_model(student_model_no_kd, (3, IMG_SIZE, IMG_SIZE), input_device=DEVICE, out_file='./images/student_architecture')

	teacher_params = sum(p.numel() for p in teacher_model.parameters())
	student_params = sum(p.numel() for p in student_model_no_kd.parameters())
	print(f'\nTeacher parameters: {teacher_params:,}')
	print(f'Student parameters: {student_params:,} ({round(student_params / teacher_params, 2)}x)')

	# Load models, or train if they don't exist

	teacher_model_path = './teacher_model.pth'
	student_model_no_kd_path = './student_model_no_kd.pth'
	student_model_with_kd_path = './student_model_with_kd.pth'

	if os.path.exists(teacher_model_path):
		teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=DEVICE))
	else:
		print('\n----- TRAINING TEACHER -----\n')
		train(teacher_model, teacher_model_path)

	if os.path.exists(student_model_no_kd_path):
		student_model_no_kd.load_state_dict(torch.load(student_model_no_kd_path, map_location=DEVICE))
	else:
		print('\n----- TRAINING STUDENT (NO KD) -----\n')
		train(student_model_no_kd, student_model_no_kd_path)

	if os.path.exists(student_model_with_kd_path):
		student_model_with_kd.load_state_dict(torch.load(student_model_with_kd_path, map_location=DEVICE))
	else:
		print('\n----- TRAINING STUDENT (WITH KD) -----\n')
		train_student_with_kd(teacher_model, student_model_with_kd, student_model_with_kd_path)

	# Test teacher, student without KD, student with KD

	print('\n----- TESTING TEACHER -----')
	test(teacher_model, 'Teacher model test')

	print('\n----- TESTING STUDENT (NO KD) -----')
	test(student_model_no_kd, 'Student model (no KD) test')

	print('\n----- TESTING STUDENT (WITH KD) -----')
	test(student_model_with_kd, 'Student model (with KD) test')
