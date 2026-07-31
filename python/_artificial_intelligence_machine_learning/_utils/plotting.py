"""
Plotting utility

Author: Sam Barba
Created 2024-03-26
"""

import cv2 as cv
# from keras.layers import Conv2D  # TensorFlow currently unavailable in Python 3.14
# from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
# import tensorflow as tf
import torch

from _utils.torchview_modded import draw_graph  # Modified version of https://github.com/mert-kurttutan/torchview


INTERPOLATION_DICT = {'nearest': cv.INTER_NEAREST, 'cubic': cv.INTER_CUBIC, 'area': cv.INTER_AREA}


def plot_image_grid(
		images, rows, cols, padding, transform=None, scale_factor=1, scale_interpolation='nearest',
		outer_padding=20, background_rgb=(128, 176, 240), title_rgb=(0, 0, 0),
		title='', save_path='', show=True
	):
	"""
	Plot a grid of images.

	Args:
		images (Iterable):
			Iterable of images (grayscale or colour; NumPy arrays or PyTorch tensors).
		rows (int):
			Number of rows to use in grid plot.
		cols (int):
			Number of columns to use in grid plot.
		padding (int):
			Size (pixels) of gap between images.
		transform (Callable):
			Optional transform to apply to each image.
		scale_factor (float):
			Scale factor to apply to each image (default: ``1``, i.e. don't scale).
		scale_interpolation (str):
			The type of interpolation to use for scaling (default: ``nearest``).
		outer_padding (int):
			Size (pixels) of border around image grid.
		background_rgb (Sequence[int]):
			R, G, B values for plot background (default: ``128, 176, 240``).
		title_rgb (Sequence[int]):
			R, G, B values for plot title (default: ``0, 0, 0``).
		title (str):
			Optional plot title.
		save_path (str):
			Optional path to save plot (if empty, plot image won't save).
		show (bool):
			Whether to show the plotted image grid.
	"""

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
		if transform:
			img = transform(img)

		img = img.squeeze()

		# Convert to numpy
		if isinstance(img, torch.Tensor):
			img = img.detach().cpu().numpy()
			if img.ndim == 3 and img.shape[0] == 3:
				img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)

		# Scale to [0,1] then convert to range [0,255]
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
		x = (output_width - title_width) // 2  # Horizontal centre
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


def plot_torch_model(model, *input_shapes, dtypes=None, device='cpu', out_file='./images/model_architecture'):
	"""
	Plot the architecture of a PyTorch model.

	Args:
		model (torch.nn.Module):
			The model whose architecture to plot.
		input_shapes (Sequence[tuple]):
			Each model input shape (excluding batch size), e.g. ``(3, 32, 32), (1,)``.
		dtypes (Sequence[torch.dtype]):
			Optional data type of each model input (defaults to ``torch.float`` if not provided).
		device (str):
			The device to use when performing forward pass on the model (default: ``cpu``).
		out_file (str):
			Where to save plot PNG (default: ``./images/model_architecture``).
	"""

	if dtypes is None:
		dtypes = [torch.float] * len(input_shapes)

	# Add batch size of 1
	x = [torch.zeros(1, *shape, device=device, dtype=dtype) for shape, dtype in zip(input_shapes, dtypes)]

	g = draw_graph(model, input_data=x)
	g.render(out_file, view=True, cleanup=True, format='png')


def plot_gradcam(model, input_img, labels=None, image_plot_transform=None):
	"""
	Plot a gradient-weighted Class Acivation Mapping (grad-CAM) heatmap for a model's predicted class.
	The grad-CAM heatmap is computed using the model's last ``torch.nn.Conv2d`` layer.

	Args:
		model (torch.nn.Module):
			The model to evaluate (must contain at least one ``torch.nn.Conv2d`` layer).
		input_img (torch.Tensor):
			Input image with shape ``(C, H, W)`` (passed to the model with batch dimension added).
		labels (Sequence[str]):
			Optional output class names (if provided, predicted class name will be added to plot title).
		image_plot_transform (torchvision.transforms):
			Optional transform to apply to input image before displaying (e.g. to undo input normalisation).
	"""

	activations = gradients = None

	# Forward hook: save feature maps
	def save_activation(module, input, output):
		nonlocal activations
		activations = output.detach()

	# Backward hook: save gradients
	def save_gradient(module, input, output):
		nonlocal gradients
		gradients = output[0].detach()


	# Find target layer (last Conv2d layer)
	last_conv = last_conv_name = None

	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Conv2d):
			last_conv = module
			last_conv_name = name

	# Register hooks
	forward_handle = last_conv.register_forward_hook(save_activation)
	backward_handle = last_conv.register_full_backward_hook(save_gradient)

	try:
		# Forward pass
		output = model(input_img.unsqueeze(dim=0))

		# Predicted class and confidence
		probabilities = torch.softmax(output, dim=1)
		pred_label = output.argmax(dim=1).item()
		pred_class = pred_label if labels is None else labels[pred_label]
		confidence = probabilities[0, pred_label].item()

		# Clear any existing gradients from previous backward passes (ensure only the image contributes to the gradients)
		model.zero_grad()

		# Backward pass for output class score
		score = output[0, pred_label]
		score.backward()

		# Remove batch dimension
		activations = activations[0]
		gradients = gradients[0]

		# Global average pooling of gradients
		weights = gradients.mean(dim=(1, 2))

		# Weighted combination of activation maps
		cam = (weights[:, None, None] * activations).sum(dim=0)

		# Keep only positive contributions
		cam = torch.nn.functional.relu(cam)

		# Scale to [0,1]
		cam = (cam - cam.min()) / (cam.max() + 1e-9)

		# Resize to input image size
		cam = torch.nn.functional.interpolate(cam[None, None], size=input_img.shape[-2:], mode='bilinear')

		_, (ax_input, ax_heatmap) = plt.subplots(ncols=2, figsize=(8, 4))

		if image_plot_transform is not None:
			ax_input.imshow(image_plot_transform(input_img.cpu()))
		elif input_img.ndim == 3 and input_img.shape[0] == 3:
			ax_input.imshow(input_img.permute(1, 2, 0).cpu())  # (C, H, W) -> (H, W, C)
		else:
			ax_input.imshow(input_img.squeeze().cpu(), cmap='gray')

		ax_input.set_title('Input image')
		ax_input.axis('off')
		ax_heatmap.imshow(cam.squeeze().cpu(), cmap='jet')
		ax_heatmap.set_title(f'Grad-CAM (layer: {last_conv_name})\nPred class = {pred_class} ({confidence:.1%})')
		ax_heatmap.axis('off')
		plt.show()
	finally:
		# Remove hooks so they don't accumulate
		forward_handle.remove()
		backward_handle.remove()


def plot_occlusion_sensitivity(model, input_img, patch_size, stride, baseline=0, labels=None, image_plot_transform=None):
	"""
	Plot an occlusion sensitivity heatmap for a model's predicted class. The heatmap is computed by sliding a square
	patch across the image, replacing each region with the baseline value, and measuring the resulting decrease in the
	model's confidence for its original predicted class.

	Args:
		model (torch.nn.Module):
			The model to evaluate.
		input_img (torch.Tensor):
			Input image with shape ``(C, H, W)`` (passed to the model with batch dimension added).
		patch_size (int):
			Height and width of the square occlusion patch.
		stride (int):
			Number of pixels to move the patch at each step.
		baseline (float | torch.Tensor):
			Value used to replace the occluded region (default: ``0``).
		labels (Sequence[str]):
			Optional output class names (if provided, predicted class name will be added to plot title).
		image_plot_transform (torchvision.transforms):
			Optional transform to apply to input image before displaying (e.g. to undo input normalisation).
	"""

	# Original prediction
	with torch.inference_mode():
		output = model(input_img.unsqueeze(dim=0))

	probabilities = torch.softmax(output, dim=1)
	pred_label = output.argmax(dim=1).item()
	pred_class = pred_label if labels is None else labels[pred_label]
	original_confidence = probabilities[0, pred_label].item()

	_, h, w = input_img.shape

	# Include the final possible patch position even when image dimensions are not exact multiples of the stride
	y_positions = list(range(0, max(h - patch_size + 1, 1), stride))
	x_positions = list(range(0, max(w - patch_size + 1, 1), stride))

	final_y = max(h - patch_size, 0)
	final_x = max(w - patch_size, 0)

	if y_positions[-1] != final_y:
		y_positions.append(final_y)
	if x_positions[-1] != final_x:
		x_positions.append(final_x)

	# Stores the confidence drop at each patch location
	occlusion_map = torch.zeros(len(y_positions), len(x_positions), device=input_img.device)

	for row, y in enumerate(y_positions):
		for col, x in enumerate(x_positions):
			# Copy the image and occlude one region
			occluded_img = input_img.clone()
			y_end = min(y + patch_size, h)
			x_end = min(x + patch_size, w)
			occluded_img[:, y:y_end, x:x_end] = baseline

			# Confidence for the original predicted class, given this occluded image
			with torch.inference_mode():
				occluded_output = model(occluded_img.unsqueeze(dim=0))
			occluded_probabilities = torch.softmax(occluded_output, dim=1)
			occluded_confidence = occluded_probabilities[0, pred_label].item()

			# Positive value means confidence decreased after occlusion
			confidence_drop = original_confidence - occluded_confidence
			occlusion_map[row, col] = confidence_drop

	# Resize to input image size
	occlusion_map = torch.nn.functional.interpolate(occlusion_map[None, None], size=(h, w), mode='bilinear')[0, 0]

	# Ignore negative values, as this plot focuses on regions whose occlusion reduces confidence
	occlusion_map = torch.nn.functional.relu(occlusion_map)

	# Scale to [0, 1]
	max_value = occlusion_map.max()
	if max_value > 0:
		occlusion_map /= max_value

	_, (ax_input, ax_occlusion) = plt.subplots(ncols=2, figsize=(8, 4))

	if image_plot_transform is not None:
		ax_input.imshow(image_plot_transform(input_img.cpu()))
	elif input_img.ndim == 3 and input_img.shape[0] == 3:
		ax_input.imshow(input_img.permute(1, 2, 0).cpu())  # (C, H, W) -> (H, W, C)
	else:
		ax_input.imshow(input_img.squeeze().cpu(), cmap='gray')

	ax_input.set_title('Input image')
	ax_input.axis('off')
	ax_occlusion.imshow(occlusion_map.cpu(), cmap='jet')
	ax_occlusion.set_title(f'Occlusion sensitivity\nPred class = {pred_class} ({original_confidence:.1%})')
	ax_occlusion.axis('off')
	plt.show()


def plot_saliency_map(model, input_img, labels=None, image_plot_transform=None):
	"""
	Plot a saliency map for a model's predicted class. The map is computed using the gradients of the output class score
	with respect to the input image. The maximum gradient magnitude across the input channels is used to create the map.

	Args:
		model (torch.nn.Module):
			The model to evaluate.
		input_img (torch.Tensor):
			Input image with shape ``(C, H, W)`` (passed to the model with batch dimension added).
		labels (Sequence[str]):
			Optional output class names (if provided, predicted class name will be added to plot title).
		image_plot_transform (torchvision.transforms):
			Optional transform to apply to input image before displaying (e.g. to undo input normalisation).
	"""

	# Enable gradients for input
	input_img.requires_grad = True

	# Forward pass
	output = model(input_img.unsqueeze(dim=0))

	# Predicted class and confidence
	probabilities = torch.softmax(output, dim=1)
	pred_label = output.argmax(dim=1).item()
	pred_class = pred_label if labels is None else labels[pred_label]
	confidence = probabilities[0, pred_label].item()

	# Clear any existing gradients from previous backward passes (ensure only the image contributes to the gradients)
	model.zero_grad()

	# Backward pass for output class score
	score = output[0, pred_label]
	score.backward()

	# Get gradients w.r.t input
	saliency = input_img.grad.data.abs()

	# Take max over the channel dimension
	saliency, _ = torch.max(saliency, dim=0)

	input_img = input_img.detach().cpu()

	_, (ax_input, ax_saliency) = plt.subplots(ncols=2, figsize=(8, 4))

	if image_plot_transform is not None:
		ax_input.imshow(image_plot_transform(input_img))
	elif input_img.ndim == 3 and input_img.shape[0] == 3:
		ax_input.imshow(input_img.permute(1, 2, 0))  # (C, H, W) -> (H, W, C)
	else:
		ax_input.imshow(input_img.squeeze(), cmap='gray')

	ax_input.set_title('Input image')
	ax_input.axis('off')
	ax_saliency.imshow(saliency.cpu(), cmap='inferno')
	ax_saliency.set_title(f'Saliency map\nPred class = {pred_class} ({confidence:.1%})')
	ax_saliency.axis('off')
	plt.show()


def get_cnn_learned_filters(conv_model):
	"""
	Extract the learned convolutional filters from a CNN. This function finds all conv layers in a PyTorch or TensorFlow
	model and computes a 2D representation of each filter by averaging its weights across the input channel dimension.

	Args:
		conv_model (torch.nn.Module | tensorflow.keras.Model):
			The CNN from which to extract learned filters.

	Returns:
		list[list[torch.Tensor | numpy.ndarray]]:
			learned filters grouped by convolutional layer. Each outer-list element corresponds to a conv layer, and
			each inner-list element contains the channel-averaged 2D filter for one output filter. PyTorch models return
			PyTorch tensors; TensorFlow models return NumPy arrays.
	"""

	# assert isinstance(conv_model, (torch.nn.Module, tf.keras.Model))
	assert isinstance(conv_model, torch.nn.Module)

	is_torch_model = isinstance(conv_model, torch.nn.Module)

	# conv_layers = [layer for layer in conv_model.modules() if isinstance(layer, torch.nn.Conv2d)] \
	# 	if is_torch_model else \
	# 	[layer for layer in conv_model.layers if isinstance(layer, Conv2D)]
	conv_layers = [layer for layer in conv_model.modules() if isinstance(layer, torch.nn.Conv2d)]

	layer_filters = []

	for idx, layer in enumerate(conv_layers, start=1):
		filters, biases = (layer.weight, layer.bias) if is_torch_model else layer.get_weights()
		num_filters = filters.shape[0 if is_torch_model else -1]

		print(f'Conv layer {idx}/{len(conv_layers)}'
			f' | Filters shape: {tuple(filters.shape)}'
			f' | Biases shape: {tuple(biases.shape)}')

		channel_means = [
			filters[i].mean(0) if is_torch_model else filters[..., i].mean(2)
			for i in range(num_filters)
		]

		layer_filters.append(channel_means)

	return layer_filters


def get_cnn_feature_maps(conv_model, input_img):
	"""
	Extract feature maps from all convolutional layers of a CNN. This function passes an input image through each conv
	layer of a PyTorch or TensorFlow model and extracts the resulting activation maps. For each conv layer, the
	individual feature map channels are returned separately.

	Args:
		conv_model (torch.nn.Module | tensorflow.keras.Model):
			The CNN from which to extract feature maps.
		input_img (torch.Tensor | numpy.ndarray):
			Input image with shape ``(C, H, W)`` for PyTorch model or ``(H, W, C)`` for TensorFlow model (passed to the
			model with batch dimension added).

	Returns:
		list[list[torch.Tensor | numpy.ndarray]]:
			feature maps grouped by convolutional layer. Each outer-list element corresponds to a conv layer, and each
			inner-list element contains a single 2D feature map channel from that layer.
	"""

	# assert isinstance(conv_model, (torch.nn.Module, tf.keras.Model))
	assert isinstance(conv_model, torch.nn.Module)

	is_torch_model = isinstance(conv_model, torch.nn.Module)

	def get_feature_map(layer):
		def hook_func(module, input, output):
			feature_maps.append(output)

		feature_maps = []  # To store the feature map
		hook_handle = layer.register_forward_hook(hook_func)
		_ = conv_model(input_img.unsqueeze(dim=0))  # Pass the input through the model
		hook_handle.remove()  # Remove hook after use

		return feature_maps[0]  # Return the feature map of the layer


	if is_torch_model:
		conv_layers = [layer for layer in conv_model.modules() if isinstance(layer, torch.nn.Conv2d)]
		feature_maps = [get_feature_map(layer) for layer in conv_layers]
	else:
		# conv_layers = [layer for layer in conv_model.layers if isinstance(layer, Conv2D)]
		# outputs = [layer.output for layer in conv_layers]
		# short_model = Model(inputs=conv_model.inputs, outputs=outputs)
		# feature_maps = short_model.predict(np.expand_dims(input_img, 0), verbose=0)
		return

	layer_feature_maps = []

	for idx, feature_map in enumerate(feature_maps, start=1):
		print(f'Feature map {idx}/{len(feature_maps)} shape: {tuple(feature_map.shape)}')

		map_depth = feature_map.shape[1 if is_torch_model else -1]  # No. channels

		feature_map_slices = [
			feature_map[0, i] if is_torch_model else feature_map[0, ..., i]
			for i in range(map_depth)
		]

		layer_feature_maps.append(feature_map_slices)

	return layer_feature_maps


def plot_confusion_matrix(y_test, y_test_pred, class_names, title, x_ticks_rotation=0, horiz_alignment='center'):
	cm = confusion_matrix(y_test, y_test_pred)
	ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(cmap='Blues')
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
