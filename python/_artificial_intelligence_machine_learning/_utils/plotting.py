"""
Plotting utility

Author: Sam Barba
Created 26/03/2024
"""

import cv2 as cv
from keras.layers import Conv2D
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
import torch

from _utils.torchview_modded import draw_graph  # Modified version of https://github.com/mert-kurttutan/torchview


INTERPOLATION_DICT = {'nearest': cv.INTER_NEAREST, 'cubic': cv.INTER_CUBIC, 'area': cv.INTER_AREA}


def plot_image_grid(
		images, rows, cols, padding, scale_factor=1, scale_interpolation='nearest',
		outer_padding=20, background_rgb=(128, 176, 240), title_rgb=(0, 0, 0),
		title='', save_path='', show=True
	):
	assert len(images) == rows * cols
	assert scale_interpolation in INTERPOLATION_DICT

	if title:
		font = ImageFont.truetype('arial.ttf', 18)
		title_width = font.getbbox(title)[2]
		# title_padding = int(1.8 * title_height)
		title_padding = 34
	else:
		title_width = title_padding = 0
		font = None

	if isinstance(images, (torch.Tensor, np.ndarray)):
		images = list(images)

	for idx, img in enumerate(images):
		# Squeeze image
		img = img.squeeze()

		# Convert to numpy
		if isinstance(img, torch.Tensor):
			img = img.detach().cpu().numpy()
			if img.ndim == 3 and img.shape[0] == 3:
				img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)

		# Normalise to [0,1] then convert to range [0,255]
		if img.max() == img.min():
			normalised_img = np.zeros_like(img)
		else:
			normalised_img = (img - img.min()) / (img.max() - img.min())
		img = (normalised_img * 255).round().astype(np.uint8)

		# Ensure image has 3 colour channels
		if img.ndim == 2:
			img = np.stack([img] * 3, axis=-1)

		# Scale by scale_factor
		if scale_factor != 1:
			h, w = img.shape[:2]
			new_h, new_w = int(h * scale_factor), int(w * scale_factor)
			img = cv.resize(img, (new_w, new_h), interpolation=INTERPOLATION_DICT[scale_interpolation])

		# Convert RGB to BGR for OpenCV
		img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

		images[idx] = img

	# Create blank output image
	img_h, img_w = images[0].shape[:2]
	output_height = rows * img_h + (rows - 1) * padding + 2 * outer_padding + title_padding
	output_width = cols * img_w + (cols - 1) * padding + 2 * outer_padding
	w_diff = 0
	if output_width - title_width < 2 * outer_padding:
		w_diff = 2 * outer_padding - output_width + title_width
		output_width += w_diff
	output_img = np.full((output_height, output_width, 3), background_rgb[::-1], dtype=np.uint8)  # RGB -> BGR

	for y in range(rows):
		for x in range(cols):
			img_idx = y * cols + x
			img_y_pos = y * (img_h + padding) + outer_padding + title_padding
			img_x_pos = x * (img_w + padding) + outer_padding + w_diff // 2
			output_img[
				img_y_pos:img_y_pos + img_h,
				img_x_pos:img_x_pos + img_w
			] = images[img_idx]

	if title:
		pil_img = Image.fromarray(output_img)
		draw = ImageDraw.Draw(pil_img)
		x = (output_width - title_width) // 2  # Horizontal center
		y = title_padding // 2 - 2
		draw.text((x, y), title, font=font, fill=title_rgb)
		output_img = np.array(pil_img)

	if save_path:
		cv.imwrite(save_path, output_img)
	if show:
		win_title = title if title else 'img'
		cv.imshow(win_title, output_img)
		cv.moveWindow(win_title, 93, 100)
		cv.waitKey(0)
		cv.destroyAllWindows()


def plot_torch_model(model, *input_shapes, input_device='cpu', out_file='./images/model_architecture'):
	# Add batch size of 1
	x = [torch.zeros((1, *shape), device=input_device) for shape in input_shapes]

	g = draw_graph(model, input_data=x)
	g.render(out_file, view=True, cleanup=True, format='png')


def get_cnn_learned_filters(conv_model, model_type='pytorch'):
	assert model_type in ('pytorch', 'tensorflow')

	conv_layers = [layer for layer in conv_model.modules() if isinstance(layer, torch.nn.Conv2d)] \
		if model_type == 'pytorch' else \
		[layer for layer in conv_model.layers if isinstance(layer, Conv2D)]

	layer_filters = []

	for idx, layer in enumerate(conv_layers, start=1):
		filters, biases = (layer.weight, layer.bias) if model_type == 'pytorch' else layer.get_weights()
		num_filters = filters.shape[0 if model_type == 'pytorch' else -1]

		print(f'Conv layer {idx}/{len(conv_layers)}'
			f' | Filters shape: {tuple(filters.shape)}'
			f' | Biases shape: {tuple(biases.shape)}')

		channel_means = [
			filters[i].mean(0) if model_type == 'pytorch' else filters[..., i].mean(2)
			for i in range(num_filters)
		]

		layer_filters.append(channel_means)

	return layer_filters


def get_cnn_feature_maps(conv_model, input_img, model_type='pytorch'):
	assert model_type in ('pytorch', 'tensorflow')

	def get_feature_map(layer):
		def hook_func(module, input, output):
			feature_maps.append(output)

		feature_maps = []  # To store the feature map
		hook_handle = layer.register_forward_hook(hook_func)
		_ = conv_model(input_img.unsqueeze(dim=0))  # Pass the input through the model
		hook_handle.remove()  # Remove hook after use

		return feature_maps[0]  # Return the feature map of the layer


	if model_type == 'pytorch':
		conv_layers = [layer for layer in conv_model.modules() if isinstance(layer, torch.nn.Conv2d)]
		feature_maps = [get_feature_map(layer) for layer in conv_layers]
	else:
		conv_layers = [layer for layer in conv_model.layers if isinstance(layer, Conv2D)]
		outputs = [layer.output for layer in conv_layers]
		short_model = Model(inputs=conv_model.inputs, outputs=outputs)
		feature_maps = short_model.predict(np.expand_dims(input_img, 0), verbose=0)

	layer_feature_maps = []

	for idx, feature_map in enumerate(feature_maps, start=1):
		print(f'Feature map {idx}/{len(feature_maps)} shape: {tuple(feature_map.shape)}')

		map_depth = feature_map.shape[1 if model_type == 'pytorch' else -1]  # No. channels

		feature_map_slices = [
			feature_map[0, i] if model_type == 'pytorch' else feature_map[0, ..., i]
			for i in range(map_depth)
		]

		layer_feature_maps.append(feature_map_slices)

	return layer_feature_maps


def plot_confusion_matrix(y_test, test_pred_labels, labels, title, x_ticks_rotation=0, horiz_alignment='center'):
	cm = confusion_matrix(y_test, test_pred_labels)
	ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(cmap='Blues')
	plt.xticks(rotation=x_ticks_rotation, ha=horiz_alignment)
	plt.title(title)
	plt.show()


def plot_roc_curve(y_test, test_pred_probs, title='ROC curve'):
	fpr, tpr, _ = roc_curve(y_test, test_pred_probs)
	plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
	plt.plot(fpr, tpr)
	plt.axis('scaled')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(title)
	plt.show()
