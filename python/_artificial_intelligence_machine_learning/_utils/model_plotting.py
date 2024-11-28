"""
Model plotting functionality

Author: Sam Barba
Created 26/03/2024
"""

from keras.layers import Conv2D
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
import torch

from _utils.torchview_modded import draw_graph  # Modified version of https://github.com/mert-kurttutan/torchview


def plot_torch_model(model, *input_shapes, input_device='cpu', out_file='./images/model_architecture'):
	# Add batch size of 1
	x = [torch.zeros((1, *shape), device=input_device) for shape in input_shapes]

	g = draw_graph(model, input_data=x)
	g.render(out_file, view=True, cleanup=True, format='png')


def plot_cnn_learned_filters(conv_model, num_cols, model_type='pytorch', title_append='', figsize=(9, 5)):
	assert model_type in ('pytorch', 'tensorflow')

	conv_layers = [layer for layer in conv_model.modules() if isinstance(layer, torch.nn.Conv2d)] \
		if model_type == 'pytorch' else \
		[layer for layer in conv_model.layers if isinstance(layer, Conv2D)]

	for idx, layer in enumerate(conv_layers, start=1):
		filters, biases = (layer.weight, layer.bias) if model_type == 'pytorch' else layer.get_weights()
		num_filters = filters.shape[0 if model_type == 'pytorch' else -1]
		num_rows = num_filters // num_cols

		print(f'Conv layer {idx}/{len(conv_layers)} | Filters shape: {tuple(filters.shape)} | Biases shape: {tuple(biases.shape)}')

		_, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)
		for ax_idx, ax in enumerate(axes.flatten()):
			# Mean of this filter across the channel dimension (0 for pytorch, 2 for tensorflow)
			channel_mean = filters[ax_idx].mean(0).detach().cpu() \
				if model_type == 'pytorch' else \
				filters[..., ax_idx].mean(2)
			ax.imshow(channel_mean, cmap='gray')
			ax.axis('off')
		plt.suptitle(f'Filters of conv layer {idx}/{len(conv_layers)}{title_append}', x=0.512, y=0.95)
		plt.gcf().set_facecolor('#80b0f0')
		plt.show()


def plot_cnn_feature_maps(conv_model, num_cols, input_img, model_type='pytorch', title_append='', figsize=(9, 5)):
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

	for idx, feature_map in enumerate(feature_maps, start=1):
		print(f'Feature map {idx}/{len(feature_maps)} shape: {tuple(feature_map.shape)}')

		map_depth = feature_map.shape[1 if model_type == 'pytorch' else -1]  # No. channels
		num_rows = map_depth // num_cols

		_, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)
		for ax_idx, ax in enumerate(axes.flatten()):
			# Plot feature_map of depth 'ax_idx'
			feature_map_slice = feature_map[0, ax_idx].detach().cpu() \
				if model_type == 'pytorch' else \
				feature_map[0, ..., ax_idx]
			ax.imshow(feature_map_slice, cmap='gray')
			ax.axis('off')
		plt.suptitle(f'Feature map of conv layer {idx}/{len(feature_maps)}{title_append}', x=0.512, y=0.95)
		plt.gcf().set_facecolor('#80b0f0')
		plt.show()


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
