"""
Demo of Adversarial Attacks on Pretrained ResNet18

Author: Sam Barba
Created 2026-06-16
"""

import torch

from _utils.plotting import plot_torch_model
from utils import *


if __name__ == '__main__':
	# Load data

	*_, test_loader = create_data_loaders()

	# Evaluate model (no adversarial attacks)

	# plot_torch_model(model, (3, img_size, img_size), device=DEVICE, out_file='./images/resnet18_architecture')
	model.eval()
	evaluate(model, test_loader)

	# Test patch attacks

	# plot_learned_patches(['untargeted'] + CLASS_SUBSET[:6], PATCH_SIZES)
	# plot_learned_patches(CLASS_SUBSET[6:], PATCH_SIZES)
	#
	# for c in ['untargeted'] + CLASS_SUBSET:
	# 	for patch_size in PATCH_SIZES:
	# 		patch_logits = torch.load(f'./artefacts/patch_{c}_{patch_size}x{patch_size}.pth', map_location=DEVICE)
	# 		evaluate(
	# 			model,
	# 			test_loader,
	# 			patch_logits=patch_logits,
	# 			target_class_name=None if c == 'untargeted' else clean_label(c)
	# 		)

	# Test perturbation attacks

	for c in ['untargeted'] + CLASS_SUBSET:
		noise_logits = torch.load(f'./artefacts/perturbation_{c}.pth', map_location=DEVICE)
		evaluate(
			model,
			test_loader,
			noise_logits=noise_logits,
			target_class_name=None if c == 'untargeted' else clean_label(c)
		)
